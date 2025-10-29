from datetime import datetime, timedelta
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.executors.pool import ThreadPoolExecutor
from apscheduler.triggers.cron import CronTrigger
import logging

from app.db.engine_default import session_scope
from app.models.models import UserDocument, Document, Organization, FileUpload, User
from app.services.document_service import delete_sentences_in_elasticsearch, delete_file_in_drive, process_upload, \
    check_orgnization_open_ai, process_upload_by_openai_test

executors = {
    'default': ThreadPoolExecutor(20),  # Increase the number of threads in the pool
    'cron_jobs': ThreadPoolExecutor(5)
}

scheduler_cron = AsyncIOScheduler(executors=executors)
logging.basicConfig(level=logging.INFO, filename='file_uploads_processing.log')


async def delete_old_documents():
    with session_scope() as session:
        thirty_days_ago = datetime.now() - timedelta(days=30)
        old_user_documents = session.query(UserDocument).join(Document, UserDocument.document_id == Document.id).filter(
            UserDocument.move_to_trash_at < thirty_days_ago).all()

        for user_document in old_user_documents:
            document = session.query(Document).filter(Document.id == user_document.document_id).first()
            if document:
                organization = session.query(Organization).filter(Organization.id == document.organization_id).first()
                if organization:
                    organization.data_used = organization.data_used - document.storage_capacity
                    session.delete(user_document)
                    session.commit()
                    session.delete(document)
                    session.commit()
                    delete_sentences_in_elasticsearch(document.id)
                    delete_file_in_drive(document.file_id)


async def schedule_delete_old_documents():
    scheduler_cron.add_job(
        delete_old_documents,
        trigger=CronTrigger(hour=0, minute=0),  # Updated to 0h00
        id="delete_old_documents_job",
        timezone="Asia/Ho_Chi_Minh",
    )


def fetch_and_process_file_uploads():
    logging.info("Starting fetch_and_process_file_uploads")
    try:
        with session_scope() as session:
            record = session.query(FileUpload).first()
            if record:
                if check_orgnization_open_ai(record.user_id):
                    logging.info(f"Processing record ai {record.id}")
                    process_upload_by_openai_test(record.document_id, record.file_path, record.user_id)
                else:
                    logging.info(f"Processing record {record.id}")
                    process_upload(record.document_id, record.file_path, record.user_id)
            else:
                logging.info("No records to process")
    except Exception as e:
        logging.error(f"Error processing file uploads: {e}")
    finally:
        logging.info("Completed fetch_and_process_file_uploads")


async def schedule_file_uploads_processing():
    scheduler_cron.add_job(
        fetch_and_process_file_uploads,
        'cron',
        minute='*/1',
        id="process_file_uploads_job",
        timezone="Asia/Ho_Chi_Minh",
    )
