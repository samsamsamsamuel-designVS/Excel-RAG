"""Run with: python -m streamlit run app.py"""

import hashlib
import json

import pandas as pd
import streamlit as st
from openai import APIConnectionError, APIStatusError, APITimeoutError, AuthenticationError

from rag import Settings, answer_question, load_tables


st.set_page_config(page_title='Chat with your tables', layout='wide')
st.title('Chat with your tables')
st.caption('Upload CSV or Excel files, then ask questions in plain English. Beta version - created by samsamsamsamuel-designVS')

try:
    settings = Settings.load()
except ValueError as exc:
    st.error(f'Check your .env settings: {exc}')
    st.stop()

# Session state belongs to this browser session, so other users have separate data.
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'tables' not in st.session_state:
    st.session_state.tables = []

with st.sidebar:
    st.header('Data upload and model settings')
    uploaded = st.file_uploader('Upload files', type=['csv', 'xlsx', 'xls'],
                               accept_multiple_files=True,
                               max_upload_size=settings.max_upload_mb)
    st.caption(f'Up to {settings.max_files} files • {settings.max_upload_mb} MB total • '
               f'{settings.max_rows:,} rows total')
    st.caption(f'Model: {settings.model}')
    if not settings.api_key:
        st.warning('Add your OPENAI_API_KEY in .env to enable chat.')
    if st.button('Clear chat', width='stretch'):
        st.session_state.messages = []
        st.rerun()

# Hash bytes as well as names: changing a file clears stale chat and evidence.
files = [(file.name, file.getvalue()) for file in uploaded] if uploaded else []
fingerprint = hashlib.sha256()
for name, data in files:
    fingerprint.update(json.dumps([name, len(data)]).encode())
    fingerprint.update(data)
# Reload tables if loading limits change in .env, too.
fingerprint.update(str((settings.max_rows, settings.max_files, settings.max_tables,
                        settings.max_upload_mb)).encode())
signature = fingerprint.hexdigest()
if signature != st.session_state.get('upload_signature'):
    st.session_state.tables = []
    st.session_state.messages = []
    st.session_state.upload_error = None
    if files:
        try:
            with st.spinner('Reading your files…'):
                st.session_state.tables = load_tables(files, settings)
        except (ValueError, OSError, UnicodeError, ImportError, pd.errors.ParserError) as exc:
            st.session_state.upload_error = str(exc)
        except Exception:
            # Corrupt/encrypted workbooks can raise engine-specific exceptions.
            st.session_state.upload_error = 'Could not read this file. Check that it is a valid, unencrypted CSV or Excel workbook.'
    st.session_state.upload_signature = signature

if st.session_state.get('upload_error'):
    st.error(st.session_state.upload_error)

tables = st.session_state.tables
if tables:
    with st.expander(f'Preview data • {len(tables)} tables', expanded=False):
        selected = st.selectbox('Choose a table', range(len(tables)),
                                format_func=lambda i: f'{tables[i].name} — {tables[i].source}')
        table = tables[selected]
        st.caption(f'{len(table.frame):,} rows • {len(table.frame.columns)} columns. '
                   'Headers are normalized to SQL-friendly names.')
        st.dataframe(table.frame.head(100), hide_index=True, width='stretch', alt='Preview of uploaded table')
else:
    st.info('Upload a file to get started.”')


def render_message(message):
    """Keep each answer together with the query and rows that support it."""
    with st.chat_message(message['role']):
        st.markdown(message['content'])
        if 'sql' in message:
            with st.expander('View SQL and supporting data'):
                st.code(message['sql'], language='sql')
                st.dataframe(message['data'], hide_index=True, width='stretch', alt='Rows supporting this answer')
                if message['truncated']:
                    st.warning(f'Results were limited; {message["evidence_rows"]} rows were sent to the model. '
                               'Ask for a summary or a narrower filter for a complete answer.')


for message in st.session_state.messages:
    render_message(message)

question = st.chat_input('Ask a question about your files…',
                         disabled=not tables or not settings.api_key, max_chars=4000,
                         submit_mode='disable')
if question:
    history = st.session_state.messages.copy()
    user_message = {'role': 'user', 'content': question}
    st.session_state.messages.append(user_message)
    render_message(user_message)
    try:
        with st.spinner('Querying your tables and preparing an answer…'):
            response = answer_question(question, tables, history, settings)
        assistant_message = {'role': 'assistant', **response}
    except AuthenticationError:
        assistant_message = {'role': 'assistant', 'content': 'The API key was rejected. Check OPENAI_API_KEY in .env.'}
    except (APIConnectionError, APITimeoutError):
        assistant_message = {'role': 'assistant', 'content': 'Could not reach the API before the timeout. Check your connection and retry.'}
    except APIStatusError as exc:
        assistant_message = {'role': 'assistant', 'content': f'The API returned HTTP {exc.status_code}. '
                             'Check your model access, quota, and billing, then retry.'}
    except ValueError as exc:
        assistant_message = {'role': 'assistant', 'content': str(exc)}
    except Exception:
        assistant_message = {'role': 'assistant', 'content': 'The generated query could not run. '
                             'Try rephrasing with the column names shown in the preview, '
                             'or ask a narrower question.'}
    st.session_state.messages.append(assistant_message)
    render_message(assistant_message)
