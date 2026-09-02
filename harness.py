from __future__ import annotations

import json
import hashlib
import hmac
import os
import tempfile
import threading
import uuid
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


LIVE_LIMIT = 20
ARCHIVE_LIMIT = 100
ENTITIES = ("memory", "tasks", "skills", "reflections", "history")
_LOCK = threading.RLock()


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


class JsonStore:
    """Small atomic JSON store with bounded live files and archive segments."""

    def __init__(self, root: str | Path = "agent_data") -> None:
        self.root = Path(root)
        self.archive_root = self.root / "archive"
        self.snapshot_root = self.root / "snapshots"
        self.root.mkdir(parents=True, exist_ok=True)
        self.archive_root.mkdir(exist_ok=True)
        self.snapshot_root.mkdir(exist_ok=True)
        self._bootstrap()

    def _write(self, path: Path, data: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with _LOCK:
            fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as handle:
                    json.dump(data, handle, indent=2, ensure_ascii=False)
                    handle.write("\n")
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(tmp_name, path)
            finally:
                if os.path.exists(tmp_name):
                    os.unlink(tmp_name)

    def _read(self, path: Path, default: Any) -> Any:
        if not path.exists():
            return deepcopy(default)
        with path.open(encoding="utf-8") as handle:
            return json.load(handle)

    def _bootstrap(self) -> None:
        descriptions = {
            "memory": "Facts, preferences, decisions, and user instructions worth recalling.",
            "tasks": "Active work and recently completed tasks.",
            "skills": "Capabilities the agent can select and invoke.",
            "reflections": "Lessons captured from outcomes and failures.",
            "history": "Recent user and assistant conversation messages.",
            "state": "Current agent status and counters.",
            "actions": "Governed operations the agent is allowed to request.",
            "policies": "Rules that authorize or reject proposed actions.",
            "auth": "Hashed local access-control configuration; never loaded as model context.",
        }
        paths = {name: f"{name}.json" for name in descriptions}
        agent_map = {
            "agentId": "harness_poc",
            "name": "Harness Agent",
            "schemaVersion": 1,
            "description": "Entry-point map. Read this before loading entity data.",
            "entities": {
                name: {
                    "path": path,
                    "description": descriptions[name],
                    "loadPolicy": "always" if name in {"state", "policies"} else "when-relevant",
                }
                for name, path in paths.items()
            },
        }
        defaults: dict[str, Any] = {
            "agent-map": agent_map,
            "memory": {"entity": "memory", "items": []},
            "tasks": {"entity": "tasks", "items": []},
            "skills": {"entity": "skills", "items": []},
            "reflections": {"entity": "reflections", "items": []},
            "history": {"entity": "history", "items": []},
            "state": {"status": "idle", "activeTaskId": None, "updatedAt": now_iso()},
            "actions": {
                "actions": [
                    {"name": "memory.store", "description": "Store an explicit memory", "policy": "memory.write"},
                    {"name": "memory.search", "description": "Find relevant memories", "policy": "memory.read"},
                    {"name": "memory.update", "description": "Update a live memory", "policy": "memory.write"},
                    {"name": "memory.forget", "description": "Forget a memory by ID", "policy": "memory.delete"},
                    {"name": "task.create", "description": "Create a task", "policy": "task.write"},
                    {"name": "task.list", "description": "List tasks", "policy": "task.read"},
                    {"name": "task.complete", "description": "Complete a task by ID", "policy": "task.write"},
                    {"name": "snapshot.create", "description": "Create a recovery snapshot", "policy": "snapshot.write"},
                    {"name": "context.search", "description": "Search live and archived agent entities", "policy": "context.read"},
                    {"name": "security.enable", "description": "Enable password protection", "policy": "security.write"},
                    {"name": "security.disable", "description": "Disable password protection", "policy": "security.write"},
                ]
            },
            "policies": {
                "policies": [
                    {"name": "memory.read", "effect": "allow"},
                    {"name": "memory.write", "effect": "allow", "condition": "explicit user request"},
                    {"name": "memory.delete", "effect": "allow", "condition": "explicit ID required"},
                    {"name": "task.write", "effect": "allow"},
                    {"name": "task.read", "effect": "allow"},
                    {"name": "snapshot.write", "effect": "allow"},
                    {"name": "context.read", "effect": "allow"},
                    {"name": "security.write", "effect": "allow"},
                ]
            },
            "auth": {"enabled": False, "salt": None, "passwordHash": None, "updatedAt": now_iso()},
        }
        for name, value in defaults.items():
            path = self.root / f"{name}.json"
            if not path.exists():
                self._write(path, value)
        # Add newly introduced map entries, actions, and policies without replacing user data.
        current_map = self.read("agent-map")
        current_map.setdefault("entities", {}).update({
            name: definition
            for name, definition in agent_map["entities"].items()
            if name not in current_map.get("entities", {})
        })
        self.write("agent-map", current_map)
        for entity, key in (("actions", "actions"), ("policies", "policies")):
            current = self.read(entity)
            known = {row["name"] for row in current.get(key, [])}
            current.setdefault(key, []).extend(row for row in defaults[entity][key] if row["name"] not in known)
            self.write(entity, current)

    def read(self, entity: str) -> Any:
        return self._read(self.root / f"{entity}.json", {})

    def write(self, entity: str, value: Any) -> None:
        self._write(self.root / f"{entity}.json", value)

    def append(self, entity: str, item: dict[str, Any]) -> dict[str, Any] | None:
        if entity not in ENTITIES:
            raise ValueError(f"Entity {entity!r} is not appendable")
        doc = self.read(entity)
        doc.setdefault("items", []).append(item)
        archived = None
        if len(doc["items"]) > LIVE_LIMIT:
            archived = doc["items"].pop(0)
            self._append_archive(entity, archived)
        self.write(entity, doc)
        return archived

    def _append_archive(self, entity: str, item: dict[str, Any]) -> None:
        folder = self.archive_root / entity
        folder.mkdir(parents=True, exist_ok=True)
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        segments = sorted(folder.glob(f"{today}.*.json"))
        path = segments[-1] if segments else folder / f"{today}.001.json"
        doc = self._read(path, {"entity": entity, "createdAt": now_iso(), "items": []})
        if len(doc["items"]) >= ARCHIVE_LIMIT:
            sequence = int(path.stem.rsplit(".", 1)[-1]) + 1
            path = folder / f"{today}.{sequence:03d}.json"
            doc = {"entity": entity, "createdAt": now_iso(), "items": []}
        doc["items"].append(item)
        doc["updatedAt"] = now_iso()
        self._write(path, doc)

    def archive_files(self) -> list[Path]:
        return sorted(self.archive_root.glob("*/*.json"), reverse=True)

    def all_items(self, entity: str) -> list[dict[str, Any]]:
        items = list(self.read(entity).get("items", []))
        for path in sorted((self.archive_root / entity).glob("*.json")):
            items.extend(self._read(path, {"items": []}).get("items", []))
        return items

    def snapshot(self) -> Path:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        payload = {
            "createdAt": now_iso(),
            "entities": {path.stem: self._read(path, {}) for path in self.root.glob("*.json")},
        }
        path = self.snapshot_root / f"snapshot-{stamp}.json"
        self._write(path, payload)
        return path

    def append_history(self, role: str, content: str) -> None:
        self.append("history", {"id": new_id("msg"), "role": role, "content": content, "createdAt": now_iso()})

    def recent_history(self) -> list[dict[str, Any]]:
        return self.read("history").get("items", [])[-LIVE_LIMIT:]

    def search_all(self, query: str, entity: str = "all", limit: int = 8) -> list[dict[str, Any]]:
        names = ENTITIES if entity == "all" else (entity,)
        words = {word.strip(".,?!:;\"'").lower() for word in query.split() if len(word.strip(".,?!:;\"'")) > 2}
        ranked: list[tuple[int, dict[str, Any]]] = []
        for name in names:
            if name not in ENTITIES:
                continue
            for item in self.all_items(name):
                haystack = json.dumps(item, ensure_ascii=False).lower()
                score = sum(word in haystack for word in words)
                if score or not words:
                    ranked.append((score, {"entity": name, **item}))
        ranked.sort(key=lambda row: (row[0], row[1].get("createdAt", "")), reverse=True)
        return [item for _, item in ranked[:limit]]

    def redact_secret(self, secret: str) -> None:
        if not secret:
            return

        def redact(value: Any) -> Any:
            if isinstance(value, str):
                return value.replace(secret, "[REDACTED]")
            if isinstance(value, list):
                return [redact(item) for item in value]
            if isinstance(value, dict):
                return {key: redact(item) for key, item in value.items()}
            return value

        for entity in ("history", "memory"):
            path = self.root / f"{entity}.json"
            self._write(path, redact(self._read(path, {})))
            for archive in (self.archive_root / entity).glob("*.json"):
                self._write(archive, redact(self._read(archive, {})))


class AuthManager:
    ITERATIONS = 600_000

    def __init__(self, store: JsonStore) -> None:
        self.store = store

    def enabled(self) -> bool:
        return bool(self.store.read("auth").get("enabled"))

    def enable(self, password: str) -> None:
        if len(password) < 4:
            raise ValueError("Password must contain at least 4 characters.")
        salt = os.urandom(16)
        digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, self.ITERATIONS)
        self.store.write("auth", {
            "enabled": True,
            "salt": salt.hex(),
            "passwordHash": digest.hex(),
            "algorithm": "pbkdf2_sha256",
            "iterations": self.ITERATIONS,
            "updatedAt": now_iso(),
        })
        self.store.redact_secret(password)
        self._remove_auth_instructions()

    def verify(self, password: str) -> bool:
        auth = self.store.read("auth")
        if not auth.get("enabled"):
            return True
        try:
            actual = hashlib.pbkdf2_hmac(
                "sha256", password.encode(), bytes.fromhex(auth["salt"]), int(auth["iterations"])
            ).hex()
            return hmac.compare_digest(actual, auth["passwordHash"])
        except (KeyError, TypeError, ValueError):
            return False

    def disable(self) -> None:
        self.store.write("auth", {
            "enabled": False,
            "salt": None,
            "passwordHash": None,
            "updatedAt": now_iso(),
        })
        self._remove_auth_instructions()

    def _remove_auth_instructions(self) -> None:
        terms = ("password", "passcode", "authentication", "unlock", "lock the agent")
        for entity in ("history", "memory"):
            doc = self.store.read(entity)
            doc["items"] = [
                item for item in doc.get("items", [])
                if not any(term in json.dumps(item, ensure_ascii=False).lower() for term in terms)
            ]
            self.store.write(entity, doc)
            for archive in (self.store.archive_root / entity).glob("*.json"):
                archived = self.store._read(archive, {"entity": entity, "items": []})
                archived["items"] = [
                    item for item in archived.get("items", [])
                    if not any(term in json.dumps(item, ensure_ascii=False).lower() for term in terms)
                ]
                self.store._write(archive, archived)


class HarnessAgent:
    def __init__(self, store: JsonStore) -> None:
        self.store = store

    def _trace(self, action: str, policy: str, result: str, archived: bool = False) -> list[dict[str, str]]:
        return [
            {"step": "Intent detected", "detail": action},
            {"step": "Policy check", "detail": f"{policy}: allowed"},
            {"step": "Action result", "detail": result},
            {"step": "Archive rotation", "detail": "1 record archived" if archived else "not required"},
        ]

    def store_memory(self, content: str, kind: str = "fact") -> dict[str, Any]:
        item = {
            "id": new_id("mem"), "type": kind, "content": content.strip(),
            "source": "explicit_user_instruction", "createdAt": now_iso(), "status": "active",
        }
        archived = self.store.append("memory", item)
        return {"message": f"Remembered: {item['content']}", "trace": self._trace("memory.store", "memory.write", item["id"], bool(archived))}

    def search_memory(self, query: str) -> dict[str, Any]:
        words = {word for word in query.lower().split() if len(word) > 2}
        items = self.store.all_items("memory")
        ranked = sorted(items, key=lambda x: sum(word in x.get("content", "").lower() for word in words), reverse=True)
        matches = [x for x in ranked if not words or any(word in x.get("content", "").lower() for word in words)][:5]
        return {"matches": matches, "message": f"Found {len(matches)} matching memories.", "trace": self._trace("memory.search", "memory.read", f"{len(matches)} match(es)")}

    def update_memory(self, memory_id: str, content: str) -> dict[str, Any]:
        doc = self.store.read("memory")
        item = next((row for row in doc.get("items", []) if row["id"] == memory_id), None)
        if not item:
            raise ValueError("Only live memories can be updated; search for the current ID first.")
        item["content"] = content.strip()
        item["updatedAt"] = now_iso()
        self.store.write("memory", doc)
        return {"message": f"Updated {memory_id}.", "trace": self._trace("memory.update", "memory.write", memory_id)}

    def forget_memory(self, memory_id: str) -> dict[str, Any]:
        doc = self.store.read("memory")
        before = len(doc.get("items", []))
        doc["items"] = [row for row in doc.get("items", []) if row["id"] != memory_id]
        if len(doc["items"]) == before:
            raise ValueError("Memory not found in the live set.")
        self.store.write("memory", doc)
        return {"message": f"Forgot {memory_id}.", "trace": self._trace("memory.forget", "memory.delete", memory_id)}

    def create_task(self, title: str) -> dict[str, Any]:
        item = {"id": new_id("task"), "title": title.strip(), "status": "active", "createdAt": now_iso()}
        archived = self.store.append("tasks", item)
        return {"message": f"Task created: {item['title']} ({item['id']})", "trace": self._trace("task.create", "task.write", item["id"], bool(archived))}

    def list_tasks(self, status: str = "all") -> dict[str, Any]:
        items = self.store.read("tasks").get("items", [])
        matches = items if status == "all" else [row for row in items if row["status"] == status]
        return {"tasks": matches, "message": f"Found {len(matches)} tasks.", "trace": self._trace("task.list", "task.read", f"{len(matches)} task(s)")}

    def complete_task(self, task_id: str) -> dict[str, Any]:
        doc = self.store.read("tasks")
        item = next((row for row in doc.get("items", []) if row["id"] == task_id), None)
        if not item:
            raise ValueError("Task not found in the live set.")
        item.update({"status": "completed", "completedAt": now_iso()})
        self.store.write("tasks", doc)
        return {"message": f"Completed {task_id}.", "trace": self._trace("task.complete", "task.write", task_id)}


class ActionExecutor:
    """The only component allowed to translate model tool calls into mutations."""

    ROUTES = {
        "memory_store": ("memory.store", "memory.write"),
        "memory_search": ("memory.search", "memory.read"),
        "memory_update": ("memory.update", "memory.write"),
        "memory_forget": ("memory.forget", "memory.delete"),
        "task_create": ("task.create", "task.write"),
        "task_list": ("task.list", "task.read"),
        "task_complete": ("task.complete", "task.write"),
        "snapshot_create": ("snapshot.create", "snapshot.write"),
        "context_search": ("context.search", "context.read"),
        "security_enable": ("security.enable", "security.write"),
        "security_disable": ("security.disable", "security.write"),
    }

    def __init__(self, store: JsonStore) -> None:
        self.store = store
        self.manager = HarnessAgent(store)

    def _authorize(self, tool_name: str) -> None:
        action, policy = self.ROUTES[tool_name]
        registered = {row["name"] for row in self.store.read("actions").get("actions", [])}
        allowed = {row["name"] for row in self.store.read("policies").get("policies", []) if row.get("effect") == "allow"}
        if action not in registered:
            raise PermissionError(f"Action {action} is not registered.")
        if policy not in allowed:
            raise PermissionError(f"Policy {policy} does not allow this action.")

    def execute(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        self._authorize(tool_name)
        if tool_name == "memory_store":
            content = arguments["content"].lower()
            protected = (
                "password", "passcode", "api key", "token", "system prompt",
                "developer message", "ignore previous", "ignore all", "authentication",
                "unlock", "security rule", "policy override",
            )
            if any(term in content for term in protected):
                raise PermissionError("Security and authentication instructions cannot be stored as memory.")
        routes = {
            "memory_store": lambda: self.manager.store_memory(arguments["content"], arguments["kind"]),
            "memory_search": lambda: self.manager.search_memory(arguments["query"]),
            "memory_update": lambda: self.manager.update_memory(arguments["memory_id"], arguments["content"]),
            "memory_forget": lambda: self.manager.forget_memory(arguments["memory_id"]),
            "task_create": lambda: self.manager.create_task(arguments["title"]),
            "task_list": lambda: self.manager.list_tasks(arguments["status"]),
            "task_complete": lambda: self.manager.complete_task(arguments["task_id"]),
            "snapshot_create": lambda: {"message": f"Created {self.store.snapshot().name}", "trace": self.manager._trace("snapshot.create", "snapshot.write", "success")},
            "context_search": lambda: {
                "matches": self.store.search_all(arguments["query"], arguments["entity"]),
                "message": "Searched current and archived agent context.",
                "trace": self.manager._trace("context.search", "context.read", "live and archives searched"),
            },
            "security_enable": lambda: self._enable_security(arguments["password"]),
            "security_disable": lambda: self._disable_security(),
        }
        return routes[tool_name]()

    def _enable_security(self, password: str) -> dict[str, Any]:
        AuthManager(self.store).enable(password)
        return {"message": "Done.", "trace": self.manager._trace("security.enable", "security.write", "password protection enabled")}

    def _disable_security(self) -> dict[str, Any]:
        AuthManager(self.store).disable()
        return {"message": "Done.", "trace": self.manager._trace("security.disable", "security.write", "password protection disabled")}
