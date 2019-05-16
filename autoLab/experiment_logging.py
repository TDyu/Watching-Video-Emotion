#!/user/bin/env python3
# -*- coding: utf-8 -*-
"""Record all operations of experiment.
"""
import logging.config
import json
import os


# Load logging config.
LOG_CONFIG_FILE = os.path.join(os.path.dirname(
    os.path.abspath(__file__)), 'log_config.json')
logging.config.dictConfig(json.load(open(LOG_CONFIG_FILE, 'r')))

# Create logger.
E_LOGGER = logging.getLogger('ExperimentLogger')
ANSWER_LOGGER = logging.getLogger('AnswerLogger')
# CONSOLE_LOGGER = logging.getLogger('consoleLogger')


def E_DEBUG(*message_texts):
    full_message = '\n'
    for message_text in message_texts:
        full_message += '\t' + str(message_text) + '\n'
    E_LOGGER.debug(full_message)


def E_INFO(*message_texts):
    full_message = '\n'
    for message_text in message_texts:
        full_message += '\t' + str(message_text) + '\n'
    E_LOGGER.info(full_message)


def E_WARN(*message_texts):
    full_message = '\n'
    for message_text in message_texts:
        full_message += '\t' + str(message_text) + '\n'
    E_LOGGER.warn(full_message)


def E_ERROR(*message_texts):
    full_message = '\n'
    for message_text in message_texts:
        full_message += '\t' + str(message_text) + '\n'
    E_LOGGER.error(full_message, exc_info=True)


def E_CRITICAL(*message_texts):
    full_message = '\n'
    for message_text in message_texts:
        full_message += '\t' + str(message_text) + '\n'
    E_LOGGER.critical(full_message)


def BASELINE_INFO(is_start=True):
    if is_start:
        E_LOGGER.info('Start base line.')
    else:
        E_LOGGER.info('End base line.')


def BREAK_INFO(is_start=True):
    if is_start:
        E_LOGGER.info('Start break.')
    else:
        E_LOGGER.info('End break.')


def COUNTDOWN_INFO(is_start=True):
    if is_start:
        E_LOGGER.info('Start countdown.')
    else:
        E_LOGGER.info('End countdown.')


def PLAY_INFO(video_name, video_length, video_seconds, is_start=True):
    if is_start:
        message = 'Start play %s: %s (%d)' % (
            video_name, video_length, video_seconds)
    else:
        message = 'End play %s: %s (%d)' % (
            video_name, video_length, video_seconds)
    E_LOGGER.info(message)


def RECORD_INFO(is_start=True):
    if is_start:
        E_LOGGER.info('Start recordind.')
    else:
        E_LOGGER.info('Stop recordind.')


def MEASURE_INFO(is_start=True):
    if is_start:
        E_LOGGER.info('Start measuring.')
    else:
        E_LOGGER.info('Stop measuring.')


def Q_INFO(is_start=True):
    if is_start:
        E_LOGGER.info('Start questionnaire.')
    else:
        E_LOGGER.info('Stop questionnaire.')


def ANSWER(video_name, answers):
    message = '%s: %s' % (video_name, str(answers))
    E_LOGGER.info(message)
    ANSWER_LOGGER.info(message)
