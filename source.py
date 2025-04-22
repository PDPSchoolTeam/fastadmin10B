import datetime
import typing as tp
from decimal import Decimal
from enum import Enum

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import (
    JSON, Boolean, Column, Date, DateTime, Float, ForeignKey,
    Integer, String, Table, Text, Time, select, update
)
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

from passlib.context import CryptContext

from fastadmin import (
    SqlAlchemyInlineModelAdmin, SqlAlchemyModelAdmin,
    WidgetType, action, display, fastapi_app as admin_app, register
)

# ================== DB Setup ====================
sqlalchemy_engine = create_async_engine("sqlite+aiosqlite:///./test.db", echo=True)
sqlalchemy_sessionmaker = async_sessionmaker(sqlalchemy_engine, expire_on_commit=False)

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def hash_password(password: str) -> str:
    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)


# ================== MODELS ======================
class EventTypeEnum(str, Enum):
    PRIVATE = "PRIVATE"
    PUBLIC = "PUBLIC"


class Base(DeclarativeBase):
    pass


class BaseModel(Base):
    __abstract__ = True

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    created_at: Mapped[datetime.datetime] = mapped_column(DateTime, default=datetime.datetime.utcnow)
    updated_at: Mapped[datetime.datetime] = mapped_column(DateTime, default=datetime.datetime.utcnow,
                                                          onupdate=datetime.datetime.utcnow)


user_m2m_event = Table(
    "event_participants",
    Base.metadata,
    Column("event_id", ForeignKey("event.id"), primary_key=True),
    Column("user_id", ForeignKey("user.id"), primary_key=True),
)


class User(BaseModel):
    __tablename__ = "user"
    username: Mapped[str] = mapped_column(String(255), nullable=False)
    password: Mapped[str] = mapped_column(String(255), nullable=False)
    is_superuser: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    avatar_url: Mapped[str | None] = mapped_column(String(255), nullable=True)
    events: Mapped[list["Event"]] = relationship(secondary=user_m2m_event, back_populates="participants")

    def __str__(self):
        return self.username


class Tournament(BaseModel):
    __tablename__ = "tournament"
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    events: Mapped[list["Event"]] = relationship(back_populates="tournament")

    def __str__(self):
        return self.name


class BaseEvent(BaseModel):
    __tablename__ = "base_event"
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    event: Mapped[tp.Optional["Event"]] = relationship(back_populates="base")

    def __str__(self):
        return self.name


class Event(BaseModel):
    __tablename__ = "event"
    base_id: Mapped[int | None] = mapped_column(ForeignKey("base_event.id"), nullable=True)
    base: Mapped[tp.Optional["BaseEvent"]] = relationship(back_populates="event")

    name: Mapped[str] = mapped_column(String(255), nullable=False)
    tournament_id: Mapped[int | None] = mapped_column(ForeignKey("tournament.id"), nullable=False)
    tournament: Mapped[tp.Optional["Tournament"]] = relationship(back_populates="events")
    participants: Mapped[list["User"]] = relationship(secondary=user_m2m_event, back_populates="events")

    rating: Mapped[int] = mapped_column(Integer, default=0)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    event_type: Mapped[EventTypeEnum] = mapped_column(default=EventTypeEnum.PUBLIC)
    is_active: Mapped[bool] = mapped_column(Boolean, default=False)
    start_time: Mapped[datetime.time | None] = mapped_column(Time)
    date: Mapped[datetime.date | None] = mapped_column(Date)
    latitude: Mapped[float | None] = mapped_column(Float)
    longitude: Mapped[float | None] = mapped_column(Float)
    price: Mapped[Decimal | None] = mapped_column(Float(asdecimal=True))
    json: Mapped[dict | None] = mapped_column(JSON)

    def __str__(self):
        return self.name


# ================ FastAdmin Registration =================

@register(User, sqlalchemy_sessionmaker=sqlalchemy_sessionmaker)
class UserModelAdmin(SqlAlchemyModelAdmin):
    list_display = ("id", "username", "is_superuser")
    list_display_links = ("id", "username")
    list_filter = ("id", "username", "is_superuser")
    search_fields = ("username",)
    formfield_overrides = {
        "username": (WidgetType.SlugInput, {"required": True}),
        "password": (WidgetType.PasswordInput, {"passwordModalForm": True}),
        "avatar_url": (WidgetType.Upload, {"required": False}),
    }

    async def authenticate(self, username, password):
        sessionmaker = self.get_sessionmaker()
        async with sessionmaker() as session:
            stmt = select(User).where(User.username == username)
            result = await session.scalars(stmt)
            user = result.first()
            if user and verify_password(password, user.password) and user.is_superuser:
                return user.id
            return None

    async def change_password(self, user_id, password):
        sessionmaker = self.get_sessionmaker()
        async with sessionmaker() as session:
            hashed_pw = hash_password(password)
            await session.execute(update(User).where(User.id == user_id).values(password=hashed_pw))
            await session.commit()


class EventInlineModelAdmin(SqlAlchemyInlineModelAdmin):
    model = Event


@register(Tournament, sqlalchemy_sessionmaker=sqlalchemy_sessionmaker)
class TournamentModelAdmin(SqlAlchemyModelAdmin):
    list_display = ("id", "name")
    inlines = (EventInlineModelAdmin,)


@register(BaseEvent, sqlalchemy_sessionmaker=sqlalchemy_sessionmaker)
class BaseEventModelAdmin(SqlAlchemyModelAdmin):
    pass


@register(Event, sqlalchemy_sessionmaker=sqlalchemy_sessionmaker)
class EventModelAdmin(SqlAlchemyModelAdmin):
    actions = ("make_is_active", "make_is_not_active")
    list_display = ("id", "name_with_price", "rating", "event_type", "is_active", "started")

    @action(description="Make user active")
    async def make_is_active(self, ids):
        sessionmaker = self.get_sessionmaker()
        async with sessionmaker() as session:
            await session.execute(update(Event).where(Event.id.in_(ids)).values(is_active=True))
            await session.commit()

    @action
    async def make_is_not_active(self, ids):
        sessionmaker = self.get_sessionmaker()
        async with sessionmaker() as session:
            await session.execute(update(Event).where(Event.id.in_(ids)).values(is_active=False))
            await session.commit()

    @display
    async def started(self, obj):
        return bool(obj.start_time)

    @display()
    async def name_with_price(self, obj):
        return f"{obj.name} - {obj.price}"


# ================= FastAPI App Setup =================

app = FastAPI()


@app.on_event("startup")
async def startup():
    async with sqlalchemy_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    async with sqlalchemy_sessionmaker() as session:
        exists = await session.scalar(select(User).where(User.username == "admin"))
        if not exists:
            user = User(
                username="admin",
                password=hash_password("admin"),
                is_superuser=True,
            )
            session.add(user)
            await session.commit()


app.mount("/admin/", admin_app)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


"""

export ADMIN_USER_MODEL=User
export ADMIN_USER_MODEL_USERNAME_FIELD=username
export ADMIN_SECRET_KEY=secret_key

"""
