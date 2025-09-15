from flask import Flask, render_template, request, jsonify
import threading
import re
import os
import uuid

from summarizer.meeting_transcript_summarizer import (
    process_transcript_file,
    send_email_notification,
)

app = Flask(__name__)

UPLOAD_FOLDER = "/tmp/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Job statuses stored in memory
# {job_id: {"status": "queued"/"processing"/"completed"/"failed", "message": "..."}}
job_statuses = {}


def background_job(job_id, transcript_path=None, transcript_text=None, emails=None, custom_prompt=None):
    """
    Run transcript summarization in a background thread.
    Supports both file upload (transcript_path) or pasted text (transcript_text),
    and multiple email recipients.
    """
    job_statuses[job_id]["status"] = "processing"
    try:
        summary = process_transcript_file(
            transcript_path=transcript_path,
            transcript_text=transcript_text,
            email_recipient=emails,
            custom_prompt=custom_prompt,
        )
        job_statuses[job_id]["status"] = "completed"
        job_statuses[job_id]["message"] = "Summary generated and sent via email."
    except Exception as e:
        job_statuses[job_id]["status"] = "failed"
        job_statuses[job_id]["message"] = str(e)
        # Send failure notification to all emails
        send_email_notification(emails, None, str(e))


@app.route("/")
def index():
    """Serve frontend page."""
    return render_template("index.html")


@app.route("/submit-transcript", methods=["POST"])
def submit_transcript():
    """
    Handle transcript submission.
    Supports file upload or pasted text.
    Validates multiple email addresses and handles optional custom prompts.
    """
    file = request.files.get("transcript")
    transcript_text = request.form.get("transcript_text", "").strip()
    email_input = request.form.get("email", "").strip()
    custom_prompt = request.form.get("custom_prompt", "").strip()

    # Validate email(s)
    if not email_input:
        return jsonify({"error": "Please provide at least one @example.com email address."}), 400

    emails = [e.strip() for e in email_input.split(",") if e.strip()]
    for e in emails:
        if not re.match(r".+@example\.com$", e):
            return jsonify({"error": f"Invalid email address: {e}"}), 400

    # Validate that exactly one input method is used
    if file and transcript_text:
        return jsonify({"error": "Please use only one method: upload file or paste text, not both."}), 400
    if not file and not transcript_text:
        return jsonify({"error": "Please provide a transcript via file upload or text paste."}), 400

    # Handle file upload if provided
    transcript_path = None
    if file:
        if not (file.filename.endswith(".txt") or file.filename.endswith(".docx")):
            return jsonify({"error": "Invalid file type. Only .txt and .docx are supported."}), 400
        job_id = str(uuid.uuid4())
        transcript_path = os.path.join(UPLOAD_FOLDER, f"{job_id}_{file.filename}")
        file.save(transcript_path)
    else:
        # Pasted text only
        job_id = str(uuid.uuid4())

    # Track job status
    job_statuses[job_id] = {"status": "queued", "message": "Job queued."}

    # Start background processing
    thread = threading.Thread(
        target=background_job,
        kwargs={
            "job_id": job_id,
            "transcript_path": transcript_path,
            "transcript_text": transcript_text if transcript_text else None,
            "emails": emails,
            "custom_prompt": custom_prompt,
        },
    )
    thread.start()

    return jsonify({"job_id": job_id}), 200


@app.route("/job-status/<job_id>", methods=["GET"])
def job_status(job_id):
    """Return job status for given job_id."""
    status = job_statuses.get(job_id)
    if status:
        return jsonify(status)
    return jsonify({"error": "Job not found"}), 404


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
