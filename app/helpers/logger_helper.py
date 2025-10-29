import logging
import sys
from logging.config import dictConfig

logging.basicConfig(stream=sys.stdout, level=logging.INFO)


class LoggerSimple:
    def __init__(self, *args, **kwargs):
        self.name = kwargs.get('name', 'default')
        dictConfig({
            'version': 1,
            'disable_existing_loggers': False,
            'formatters': {
                'console': {
                    'format': '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'
                },
                'file': {
                    'format': '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'
                }
            },
            'handlers': {
                'console': {
                    'class': 'logging.StreamHandler',
                    'formatter': 'console',
                },

            },
            'loggers': {
                # root logger
                '': {
                    'level': 'INFO',
                    'handlers': ['console'],
                },
                'stomp.py ': {
                    'level': 'WARN',
                    'handlers': ['console'],
                },
            },
        })
        logging.getLogger("stomp.py").setLevel(logging.WARN)
        logging.getLogger("elasticsearch").setLevel(logging.WARN)
        logging.getLogger("elastic_transport").setLevel(logging.WARN)
        logging.getLogger("pika").setLevel(logging.WARNING)
        logging.getLogger("celery.app.trace").setLevel(logging.WARNING)
        logging.getLogger("py4j").setLevel(logging.WARN)
        logging.getLogger("pyppeteer.launcher").setLevel(logging.WARN)
        # logging.getLogger('sqlalchemy.engine').setLevel(logging.INFO)
        self.logger = logging.getLogger(self.name)
