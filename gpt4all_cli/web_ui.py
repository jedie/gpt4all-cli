import html
import logging
import multiprocessing
import re
import socket
from datetime import datetime
from time import time
from uuid import uuid1, uuid4

from gpt4all import GPT4All, LLModel
from lona import App, Channel, RedirectResponse, View
from lona.channels import Message
from lona.html import H2, Option2, Select2
from lona_picocss import install_picocss
from lona_picocss.html import (
    H1,
    HTML,
    A,
    Br,
    Div,
    InlineButton,
    P,
    ScrollerDiv,
    Span,
    Strong,
    Table,
    TBody,
    Td,
    TextArea,
    TextInput,
    Th,
    THead,
    Tr,
)

from gpt4all_cli.data_classes import ChatMessage, MessageTypeEnum, RoomData, RoomState


logger = logging.getLogger(__name__)


NAME = re.compile(r'^([a-zA-Z0-9-_]{1,})$')
MESSAGE_BACK_LOG = 10
GPT_WRITE_ELLIPSIS = '\N{MIDLINE HORIZONTAL ELLIPSIS}'  # U+22EF
# GPT_WRITE_ELLIPSIS = '\N{HORIZONTAL ELLIPSIS}'  # U+2026

WELCOME_PROMPT = 'Create a nice, short welcoming message to a new visitor of this chat.'
WELCOME_MAX_TOKENS = 50
MAX_TOKENS = 100


class Gpt:
    def __init__(self, *, channel, room_data: RoomData):
        self.channel = channel
        self.room_data = room_data
        self.message_id = uuid4().hex

    def __enter__(self):
        self.room_data.state = RoomState.GPT_WRITES

        message = ChatMessage(
            id=self.message_id,
            type=MessageTypeEnum.WAIT,
            dt=datetime.now(),
            user_name='GPT',
        )
        self.channel.send(message_data={'message': message})

        return self

    def generate(self, *, prompt, max_tokens=300):
        chat_session = self.room_data.chat_session
        generator = chat_session.generate(prompt=prompt, streaming=True, max_tokens=max_tokens)
        for token in generator:
            token = token.replace('\n', ' ')
            logger.info('GPT token: %r', token)
            self.channel.send(
                message_data={
                    'message': ChatMessage(
                        id=self.message_id,
                        type=MessageTypeEnum.APPEND,
                        message=token,
                    )
                }
            )

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.room_data.state = RoomState.FREE
        self.channel.send(
            message_data={
                'message': ChatMessage(
                    id=self.message_id,
                    type=MessageTypeEnum.COMPLETE,
                )
            }
        )
        if exc_type:
            return False


class GptChatApp(App):
    pass


app = GptChatApp(__file__)

install_picocss(app)

app.settings.PICOCSS_BRAND = 'Multi-User GPT Chat'
app.settings.PICOCSS_TITLE = 'Multi-User GPT Chat'

app.settings.INITIAL_SERVER_STATE = {
    'user': {},
    'rooms': {},
}


@app.route('/<room>(/)', name='room')
class ChatView(View):
    def show_message(self, message, index=None):
        message_id, unix_timestamp, type, user_name, message = message

        span = Span(style='margin-left: 0.5em')

        line = Div(
            Div(
                Strong(user_name),
                Span(
                    str(datetime.fromtimestamp(unix_timestamp)),
                    style={
                        'color': 'gray',
                        'font-size': '75%',
                        'margin-left': '0.5em',
                    },
                ),
            ),
            span,
            data_message_id=message_id,
        )

        if type == 'message':
            span.set_text(message)

        else:
            span.set_text(f'*{message}*')

            if type == 'join':
                span.style['color'] = 'lime'

            elif type == 'leave':
                span.style['color'] = 'red'

        with self.html.lock:
            if self.messages_scroller.query_selector(f'[data-message-id={message_id}]'):
                return

            if index is None:
                self.messages_scroller.append(line)

            else:
                self.messages_scroller.insert(index, line)

            self.show(self.html)

    def handle_messages(self, message: Message):
        chat_message: ChatMessage = message.data['message']
        if isinstance(chat_message, list):
            return self.show_message(chat_message)

        message_type: MessageTypeEnum = chat_message.type

        if message_type == MessageTypeEnum.WAIT:
            line = Div(
                Div(
                    Strong(chat_message.user_name),
                    Span(
                        str(chat_message.dt),
                        style={
                            'color': 'gray',
                            'font-size': '75%',
                            'margin-left': '0.5em',
                        },
                    ),
                ),
                Span(style='margin-left: 0.5em', data_message_id=chat_message.id),
            )
            self.messages_scroller.append(line)
        elif message_type == MessageTypeEnum.APPEND:
            span = self.messages_scroller.query_selector(f'[data-message-id={chat_message.id}]')
            old_text = span.get_text()
            old_text = old_text.rstrip(GPT_WRITE_ELLIPSIS)
            new_token = chat_message.message
            new_token = html.escape(new_token)
            new_token = new_token.replace('\n', '<br>')
            span.set_text(f'{old_text}{new_token}{GPT_WRITE_ELLIPSIS}')

        elif message_type == MessageTypeEnum.COMPLETE:
            span = self.messages_scroller.query_selector(f'[data-message-id={chat_message.id}]')
            text = span.get_text()
            text = text.rstrip(GPT_WRITE_ELLIPSIS)
            if not text:
                text = html.escape('<No answer from GPT>')
            span.set_text(text.rstrip(GPT_WRITE_ELLIPSIS))

        else:
            raise NotImplementedError(f'Unknown message type: {chat_message.type}')

        with self.html.lock:
            self.show(self.html)

    def send_message(self, type, text):
        if type == 'join':
            with Gpt(channel=self.channel, room_data=self.room_data) as gpt:
                gpt.generate(prompt=WELCOME_PROMPT, max_tokens=WELCOME_MAX_TOKENS)
            return

        message = [
            uuid1().hex,
            time(),
            type,
            self.user_name,
            text,
        ]
        logger.info('send_message: %s', message)

        # add message to data
        self.room_data.logs.append(message)

        # send message to all clients
        self.channel.send({'message': message})

        # trim messages
        while len(self.room_data.logs) > MESSAGE_BACK_LOG:
            self.room_data.logs.pop(0)

        if type == 'message':
            with Gpt(channel=self.channel, room_data=self.room_data) as gpt:
                gpt.generate(prompt=text, max_tokens=MAX_TOKENS)

    def handle_send_button_click(self, input_event):
        message = self.message_text_area.value.strip()
        self.message_text_area.value = ''

        # nothing to send
        if not message:
            return

        self.send_message('message', message)

    def handle_request(self, request):
        self.room_name = request.match_info['room']
        self.session_key = request.user.session_key
        self.user_name = self.server.state['user'].get(self.session_key, '')
        self.joined = False

        # redirect to lobby if the user has no user name set
        if not self.user_name:
            return RedirectResponse(self.server.reverse('lobby'))

        # check if room exists
        if self.room_name not in self.server.state['rooms']:
            return HTML(
                H1('Room not found'),
                P(f'No room named "{self.room_name}" found'),
            )

        # setup html
        self.room_data: RoomData = self.server.state['rooms'][self.room_name]

        self.messages_scroller = ScrollerDiv(lines=MESSAGE_BACK_LOG, height='50vh')
        self.message_text_area = TextArea()

        self.send_button = InlineButton(
            'Send',
            handle_click=self.handle_send_button_click,
        )

        chat_session = self.room_data.chat_session
        model_config = chat_session.config
        model: LLModel = chat_session.model
        thread_count = model.thread_count()

        table = Table(
            THead(Tr(Th('Parameter'), Th('Value'))),
            TBody(),
        )
        table.append(Tr(Td('Thread count'), Td(str(thread_count))))
        for key, value in model_config.items():
            table.append(Tr(Td(key), Td(html.escape(repr(value)))))

        self.html = HTML(
            H1(f'Chat Room: "{self.room_name}"'),
            P(f'{model_config["type"]} - {model_config["name"]} ({model_config["filename"]})'),
            self.messages_scroller,
            self.message_text_area,
            self.send_button,
            H2('model config:'),
            table,
        )

        # subscribe to channel
        self.channel = self.subscribe(f'chat.room.{self.room_name}', self.handle_messages)

        self.room_data.users.append(self.user_name)
        self.send_message('join', 'Joined')

        # load history
        for index, message in enumerate(self.room_data.logs.copy()):
            self.show_message(message, index=index)

        self.joined = True

        return self.html

    def on_cleanup(self) -> None:
        if not self.joined:
            return

        self.room_data.users.remove(self.user_name)
        self.send_message('leave', 'Left')


@app.route('/', name='lobby')
class LobbyView(View):
    # alerts

    def show_error_alert(self, *message):
        with self.html.lock:
            self.alerts.style['color'] = 'red'
            self.alerts.nodes = list(message)

    def show_success_alert(self, *message):
        with self.html.lock:
            self.alerts.style['color'] = 'lime'
            self.alerts.nodes = list(message)

    # user name
    def set_user_name(self, input_event):
        name = self.user_name.value

        if not NAME.match(name):
            self.show_error_alert(f'"{name}" is no valid name')
            return

        if name in self.server.state['user']:
            self.show_error_alert(f'"{name}" is already taken')
            return

        self.server.state['user'][self.session_key] = name

        return RedirectResponse('.')

    # rooms
    def list_rooms(self, *args, **kwargs):
        with self.html.lock:
            self.room_table[-1].clear()

            for room_name, room_data in self.server.state['rooms'].items():
                room_data: RoomData
                user_count = len(room_data.users)

                self.room_table[-1].append(
                    Tr(
                        Td(
                            A(
                                room_name,
                                href=self.server.reverse('room', room=room_name),
                            ),
                        ),
                        Td(room_data.gpt_model_name),
                        Td(str(user_count)),
                    ),
                )

    def create_room(self, input_event):
        name = self.room_name.value
        gpt_model_name = self.gpt_model_name.value

        if not NAME.match(name):
            self.show_error_alert(f'"{name}" is no valid name')

            return

        if name in self.server.state['rooms']:
            self.show_error_alert(f'"{name}" is already taken')
            return

        logger.info('create_room: %s gpt_model_name: %s', name, gpt_model_name)
        self.show_success_alert(f'Creating {name!r} room with {gpt_model_name}...')
        gpt4all = GPT4All(model_name=gpt_model_name, n_threads=multiprocessing.cpu_count(), verbose=True)
        chat_session = gpt4all.chat_session().__enter__()

        room_data = RoomData(gpt_model_name=gpt_model_name, chat_session=chat_session)
        logger.debug('create room %r for %r with: %r', name, gpt_model_name, room_data)

        self.server.state['rooms'][name] = room_data

        self.room_name.value = ''
        self.show_success_alert(f'Room "{name}" with {gpt_model_name} was created')

        Channel('chat.room.open').send()

    def handle_request(self, request):
        self.session_key = request.user.session_key
        self.alerts = P()

        # set name
        if self.session_key not in self.server.state['user']:
            self.user_name = TextInput(placeholder='User Name', value=socket.gethostname())

            self.set_user_name_button = InlineButton(
                'Set',
                handle_click=self.set_user_name,
            )

            self.html = HTML(
                H1('Set User Name'),
                self.alerts,
                self.user_name,
                self.set_user_name_button,
            )

            return self.html

        self.gpt_model_name = Select2(
            Option2('em_german_mistral_v01.Q4_0.gguf', value='em_german_mistral_v01.Q4_0.gguf', selected=True),
            Option2('orca-mini-3b-gguf2-q4_0.gguf', value='orca-mini-3b-gguf2-q4_0.gguf'),
            Option2('wizardlm-13b-v1.2.Q4_0.gguf', value='wizardlm-13b-v1.2.Q4_0.gguf'),
            Option2('mistral-7b-openorca.Q4_0.gguf', value='mistral-7b-openorca.Q4_0.gguf'),
        )
        self.room_name = TextInput(placeholder='Room Name', value='test')

        self.create_room_button = InlineButton(
            'Create Room',
            handle_click=self.create_room,
        )

        self.room_table = Table(
            THead(
                Tr(
                    Th('Room Name'),
                    Th('GPT model'),
                    Th('User Chatting'),
                ),
            ),
            TBody(),
        )

        self.html = HTML(
            H1('Chat Rooms'),
            self.alerts,
            self.gpt_model_name,
            self.room_name,
            self.create_room_button,
            Br(),
            Br(),
            self.room_table,
        )

        self.list_rooms()

        self.channel = self.subscribe(
            'chat.room.*',
            self.list_rooms,
        )

        return self.html
