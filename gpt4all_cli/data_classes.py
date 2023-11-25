import dataclasses
import enum
from datetime import datetime

from gpt4all import GPT4All


class MessageTypeEnum(enum.StrEnum):
    WAIT = enum.auto()
    APPEND = enum.auto()
    COMPLETE = enum.auto()


class RoomState(enum.StrEnum):
    GPT_WRITES = enum.auto()
    FREE = enum.auto()


@dataclasses.dataclass
class ChatMessage:
    id: str
    type: MessageTypeEnum

    dt: datetime | None = None
    user_name: str | None = None
    message: str | None = None


@dataclasses.dataclass
class RoomData:
    gpt_model_name: str
    chat_session: GPT4All

    users: list[str] = dataclasses.field(default_factory=list)
    logs: list = dataclasses.field(default_factory=list)

    state: RoomState = RoomState.FREE
