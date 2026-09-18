"""
AI Quiz Generator
A beginner-friendly Streamlit app that creates AI-powered MCQ quizzes
using the Google Gemini API.

Run locally:
    pip install -r requirements.txt
    streamlit run app.py

For local secrets, create:
    .streamlit/secrets.toml

with:
    GEMINI_API_KEY = "your-api-key-here"

Never commit secrets.toml to GitHub.
"""

import io
import re
from datetime import datetime
from typing import Any

import pandas as pd
import streamlit as st
from google import genai
from google.genai import types
from streamlit_autorefresh import st_autorefresh


# ============================================================
# APP CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Quiz Generator",
    page_icon="🧠",
    layout="centered",
)

# Current Gemini model. You can change this later if required.
GEMINI_MODEL = "gemini-3.6-flash"

# Available timer choices in minutes.
TIMER_OPTIONS = [1, 2, 5, 10, 15, 20, 30]

# Grading thresholds are kept in one place so they are easy to edit.
GRADE_RULES = [
    (90, "A+", "Excellent"),
    (80, "A", "Very Good"),
    (70, "B", "Good"),
    (60, "C", "Satisfactory"),
    (50, "D", "Needs Improvement"),
    (0, "F", "Poor"),
]


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>
        .main-title {
            text-align: center;
            font-size: 2.4rem;
            font-weight: 700;
            margin-bottom: 0.2rem;
        }

        .subtitle {
            text-align: center;
            color: #6b7280;
            margin-bottom: 2rem;
        }

        .timer-box {
            text-align: center;
            padding: 0.8rem;
            border-radius: 12px;
            border: 1px solid #d1d5db;
            margin-bottom: 1rem;
        }

        .timer-value {
            font-size: 2rem;
            font-weight: 700;
        }

        .question-number {
            font-size: 1rem;
            color: #6b7280;
            margin-bottom: 0.3rem;
        }

        .question-text {
            font-size: 1.35rem;
            font-weight: 600;
            margin-bottom: 1rem;
        }

        .result-score {
            text-align: center;
            font-size: 2.5rem;
            font-weight: 700;
        }

        .result-grade {
            text-align: center;
            font-size: 1.8rem;
            font-weight: 700;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SESSION STATE
# ============================================================

def initialize_session_state() -> None:
    """Create all session-state variables used by the app."""

    defaults = {
        "page": "start",
        "topic": "",
        "total_questions": 10,
        "timer_minutes": 10,
        "questions": [],
        "answers": {},
        "current_question": 0,
        "quiz_started_at": None,
        "quiz_ends_at": None,
        "quiz_finished_at": None,
        "result": None,
        "quiz_start_display_time": None,
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


initialize_session_state()


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_gemini_api_key() -> str | None:
    """Safely read the Gemini API key from Streamlit Secrets."""

    try:
        api_key = st.secrets["GEMINI_API_KEY"]
        if api_key:
            return str(api_key).strip()
    except Exception:
        return None

    return None


def sanitize_filename(text: str) -> str:
    """Convert a topic into a safe filename."""

    cleaned = re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_")

    if not cleaned:
        cleaned = "quiz"

    return cleaned[:60]


def format_seconds(seconds: int) -> str:
    """Convert seconds into MM:SS format."""

    seconds = max(0, int(seconds))
    minutes = seconds // 60
    remaining_seconds = seconds % 60

    return f"{minutes:02d}:{remaining_seconds:02d}"


def get_grade(percentage: float) -> tuple[str, str]:
    """Return grade and remarks based on the percentage."""

    for minimum, grade, remarks in GRADE_RULES:
        if percentage >= minimum:
            return grade, remarks

    return "F", "Poor"


def validate_questions(data: Any, requested_count: int) -> list[dict[str, Any]]:
    """
    Validate Gemini's generated quiz.

    The function checks:
    - correct number of questions
    - question text
    - exactly 4 options
    - duplicate options
    - correct answer
    - valid answer index
    """

    if not isinstance(data, dict):
        raise ValueError("Gemini did not return a valid quiz object.")

    questions = data.get("questions")

    if not isinstance(questions, list):
        raise ValueError("The generated quiz does not contain a question list.")

    if len(questions) != requested_count:
        raise ValueError(
            f"Gemini generated {len(questions)} questions instead of "
            f"{requested_count}."
        )

    validated = []
    seen_questions = set()

    for number, item in enumerate(questions, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"Question {number} has an invalid format.")

        question_text = str(item.get("question", "")).strip()
        options = item.get("options")
        correct_answer = item.get("correct_answer")

        if not question_text:
            raise ValueError(f"Question {number} has no question text.")

        if question_text.lower() in seen_questions:
            raise ValueError(f"Duplicate question found at question {number}.")

        seen_questions.add(question_text.lower())

        if not isinstance(options, list) or len(options) != 4:
            raise ValueError(
                f"Question {number} must contain exactly 4 options."
            )

        options = [str(option).strip() for option in options]

        if any(not option for option in options):
            raise ValueError(f"Question {number} contains an empty option.")

        if len({option.lower() for option in options}) != 4:
            raise ValueError(
                f"Question {number} contains duplicate answer options."
            )

        try:
            correct_index = int(correct_answer)
        except (TypeError, ValueError):
            raise ValueError(
                f"Question {number} has an invalid correct answer."
            )

        if correct_index not in range(4):
            raise ValueError(
                f"Question {number} correct answer must be 0, 1, 2, or 3."
            )

        validated.append(
            {
                "question": question_text,
                "options": options,
                "correct_answer": correct_index,
            }
        )

    return validated


def generate_questions(topic: str, count: int) -> list[dict[str, Any]]:
    """Generate quiz questions using Gemini structured JSON output."""

    api_key = get_gemini_api_key()

    if not api_key:
        raise RuntimeError(
            "Gemini API key was not found. Add GEMINI_API_KEY to "
            ".streamlit/secrets.toml or Streamlit Cloud Secrets."
        )

    client = genai.Client(api_key=api_key)

    quiz_schema = {
        "type": "object",
        "properties": {
            "questions": {
                "type": "array",
                "minItems": count,
                "maxItems": count,
                "items": {
                    "type": "object",
                    "properties": {
                        "question": {
                            "type": "string",
                            "description": "The MCQ question.",
                        },
                        "options": {
                            "type": "array",
                            "minItems": 4,
                            "maxItems": 4,
                            "items": {"type": "string"},
                            "description": "Exactly four answer options.",
                        },
                        "correct_answer": {
                            "type": "integer",
                            "minimum": 0,
                            "maximum": 3,
                            "description": (
                                "Zero-based index of the correct option. "
                                "0=A, 1=B, 2=C, 3=D."
                            ),
                        },
                    },
                    "required": [
                        "question",
                        "options",
                        "correct_answer",
                    ],
                },
            }
        },
        "required": ["questions"],
    }

    prompt = f"""
Create a high-quality multiple-choice quiz about:

TOPIC:
{topic}

NUMBER OF QUESTIONS:
{count}

Rules:
1. Generate exactly {count} unique questions.
2. Every question must have exactly four options.
3. Options must be meaningful and plausible.
4. There must be exactly one correct option.
5. Mix easy, medium, and difficult questions where appropriate.
6. Questions must be directly relevant to the requested topic.
7. Do not use duplicate questions.
8. Do not include explanations.
9. The correct_answer field must be a zero-based integer:
   0 = first option
   1 = second option
   2 = third option
   3 = fourth option.
10. Return data matching the supplied JSON schema.
"""

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.4,
            response_mime_type="application/json",
            response_schema=quiz_schema,
        ),
    )

    if not response.text:
        raise ValueError("Gemini returned an empty response.")

    # Structured output should already be valid JSON.
    # We still parse and validate it before starting the quiz.
    import json

    data = json.loads(response.text)
    return validate_questions(data, count)


def calculate_result() -> dict[str, Any]:
    """Calculate score, percentage, grade, and answer statistics."""

    questions = st.session_state.questions
    answers = st.session_state.answers

    total = len(questions)
    correct = 0
    wrong = 0
    unanswered = 0

    review = []

    for index, question in enumerate(questions):
        user_answer = answers.get(index)
        correct_index = question["correct_answer"]

        if user_answer is None:
            result_status = "Unanswered"
            unanswered += 1
            user_answer_text = "Unanswered"
        elif user_answer == correct_index:
            result_status = "Correct"
            correct += 1
            user_answer_text = question["options"][user_answer]
        else:
            result_status = "Wrong"
            wrong += 1
            user_answer_text = question["options"][user_answer]

        correct_answer_text = question["options"][correct_index]

        review.append(
            {
                "Question Number": index + 1,
                "Question": question["question"],
                "User Answer": user_answer_text,
                "Correct Answer": correct_answer_text,
                "Result": result_status,
            }
        )

    percentage = (correct / total * 100) if total else 0
    grade, remarks = get_grade(percentage)

    return {
        "total": total,
        "correct": correct,
        "wrong": wrong,
        "unanswered": unanswered,
        "percentage": percentage,
        "grade": grade,
        "remarks": remarks,
        "review": review,
    }


def create_excel_file(result: dict[str, Any]) -> bytes:
    """Create an Excel result file in memory."""

    summary = pd.DataFrame(
        [
            ["Quiz Topic", st.session_state.topic],
            [
                "Date/Time",
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            ],
            ["Total Questions", result["total"]],
            ["Correct Answers", result["correct"]],
            ["Wrong Answers", result["wrong"]],
            ["Unanswered Questions", result["unanswered"]],
            [
                "Total Score",
                f'{result["correct"]} / {result["total"]}',
            ],
            ["Percentage", f'{result["percentage"]:.2f}%'],
            ["Grade", result["grade"]],
            ["Remarks", result["remarks"]],
        ],
        columns=["Item", "Value"],
    )

    review_df = pd.DataFrame(result["review"])

    output = io.BytesIO()

    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        summary.to_excel(
            writer,
            sheet_name="Quiz Result",
            index=False,
        )

        review_df.to_excel(
            writer,
            sheet_name="Question Review",
            index=False,
        )

        # Basic column sizing.
        for sheet in writer.book.worksheets:
            for column_cells in sheet.columns:
                max_length = 0

                for cell in column_cells:
                    value = "" if cell.value is None else str(cell.value)
                    max_length = max(max_length, len(value))

                adjusted_width = min(max(max_length + 2, 12), 60)
                sheet.column_dimensions[column_cells[0].column_letter].width = (
                    adjusted_width
                )

    output.seek(0)
    return output.getvalue()


def reset_quiz() -> None:
    """Clear quiz state and return to the start screen."""

    keys_to_reset = [
        "page",
        "topic",
        "total_questions",
        "timer_minutes",
        "questions",
        "answers",
        "current_question",
        "quiz_started_at",
        "quiz_ends_at",
        "quiz_finished_at",
        "result",
        "quiz_start_display_time",
    ]

    defaults = {
        "page": "start",
        "topic": "",
        "total_questions": 10,
        "timer_minutes": 10,
        "questions": [],
        "answers": {},
        "current_question": 0,
        "quiz_started_at": None,
        "quiz_ends_at": None,
        "quiz_finished_at": None,
        "result": None,
        "quiz_start_display_time": None,
    }

    for key in keys_to_reset:
        st.session_state[key] = defaults[key]


def finish_quiz() -> None:
    """Finish the quiz and calculate the final result."""

    if st.session_state.page == "result":
        return

    st.session_state.quiz_finished_at = datetime.now()
    st.session_state.result = calculate_result()
    st.session_state.page = "result"


# ============================================================
# START PAGE
# ============================================================

def show_start_page() -> None:
    """Display the quiz setup screen."""

    st.markdown(
        '<div class="main-title">🧠 AI Quiz Generator</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="subtitle">'
        "Create an AI-powered quiz from any topic"
        "</div>",
        unsafe_allow_html=True,
    )

    with st.container(border=True):
        st.subheader("Quiz Setup")

        topic = st.text_input(
            "1. Quiz Topic",
            placeholder="Example: Newton's Laws of Motion",
            key="topic_input",
        )

        col1, col2 = st.columns(2)

        with col1:
            total_questions = st.number_input(
                "2. Total MCQs",
                min_value=5,
                max_value=50,
                value=10,
                step=1,
            )

        with col2:
            timer_minutes = st.selectbox(
                "3. Test Timer",
                TIMER_OPTIONS,
                index=3,
                format_func=lambda x: f"{x} minute"
                if x == 1
                else f"{x} minutes",
            )

        st.write("")

        if st.button(
            "🚀 Start Quiz",
            type="primary",
            use_container_width=True,
        ):
            cleaned_topic = topic.strip()

            if not cleaned_topic:
                st.error("Please enter a quiz topic.")
                return

            with st.spinner(
                "Generating your quiz with Gemini AI..."
            ):
                try:
                    questions = generate_questions(
                        cleaned_topic,
                        int(total_questions),
                    )

                    st.session_state.topic = cleaned_topic
                    st.session_state.total_questions = int(total_questions)
                    st.session_state.timer_minutes = int(timer_minutes)
                    st.session_state.questions = questions
                    st.session_state.answers = {}
                    st.session_state.current_question = 0

                    start_time = datetime.now()
                    st.session_state.quiz_started_at = start_time
                    st.session_state.quiz_ends_at = (
                        start_time
                        + pd.Timedelta(minutes=int(timer_minutes))
                    ).to_pydatetime()

                    st.session_state.quiz_finished_at = None
                    st.session_state.quiz_start_display_time = start_time
                    st.session_state.result = None
                    st.session_state.page = "quiz"

                    st.rerun()

                except Exception as error:
                    st.error(
                        "Could not generate the quiz. "
                        "Please check your Gemini API key, internet "
                        "connection, and try again."
                    )
                    st.caption(f"Technical detail: {error}")


# ============================================================
# QUIZ PAGE
# ============================================================

def show_quiz_page() -> None:
    """Display one question at a time."""

    # Server-side timer check.
    now = datetime.now()
    end_time = st.session_state.quiz_ends_at

    if end_time is None:
        st.error("Quiz timer information is missing.")
        if st.button("Start New Quiz"):
            reset_quiz()
            st.rerun()
        return

    remaining_seconds = int((end_time - now).total_seconds())

    if remaining_seconds <= 0:
        finish_quiz()
        st.rerun()

    # Refresh the Streamlit app every second so the timer updates.
    # This package is lightweight and makes the countdown reliable.
    st_autorefresh(
        interval=1000,
        limit=None,
        key="quiz_timer_refresh",
    )

    questions = st.session_state.questions
    current = st.session_state.current_question
    total = len(questions)

    current_question = questions[current]

    # Header information.
    st.markdown(
        f"### 🧠 {st.session_state.topic}"
    )

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown(
            f"**Question {current + 1} of {total}**"
        )

    with col2:
        st.markdown(
            f"""
            <div class="timer-box">
                <div>⏱️ Time Remaining</div>
                <div class="timer-value">{format_seconds(remaining_seconds)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.progress((current + 1) / total)

    st.divider()

    st.markdown(
        f'<div class="question-text">{current_question["question"]}</div>',
        unsafe_allow_html=True,
    )

    option_labels = ["A", "B", "C", "D"]

    # The selected option is stored using a normal session-state
    # dictionary, which avoids losing answers during reruns.
    existing_answer = st.session_state.answers.get(current)

    selected_option = st.radio(
        "Select your answer:",
        options=range(4),
        format_func=lambda i: (
            f"{option_labels[i]}. {current_question['options'][i]}"
        ),
        index=existing_answer,
        key=f"question_{current}",
    )

    # Save the selected answer.
    if selected_option is not None:
        st.session_state.answers[current] = selected_option

    st.write("")

    # Navigation buttons.
    nav1, nav2, nav3 = st.columns([1, 1, 1])

    with nav1:
        if current > 0:
            if st.button(
                "⬅️ Previous",
                use_container_width=True,
            ):
                st.session_state.current_question -= 1
                st.rerun()

    with nav2:
        if current < total - 1:
            if st.button(
                "Next ➡️",
                type="primary",
                use_container_width=True,
            ):
                st.session_state.current_question += 1
                st.rerun()

    with nav3:
        if current == total - 1:
            if st.button(
                "✅ Submit Quiz",
                type="primary",
                use_container_width=True,
            ):
                finish_quiz()
                st.rerun()

    # Show answered-question status without revealing answers.
    answered_count = len(st.session_state.answers)

    st.caption(
        f"Answered: {answered_count} / {total}"
    )


# ============================================================
# RESULT PAGE
# ============================================================

def show_result_page() -> None:
    """Display the final quiz result and downloadable report."""

    result = st.session_state.result

    if not result:
        st.error("No quiz result is available.")
        return

    st.markdown(
        '<div class="main-title">🎉 Quiz Completed!</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        f'<div class="result-score">'
        f'{result["correct"]} / {result["total"]}'
        f"</div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        f'<div class="result-grade">'
        f'Grade: {result["grade"]} — {result["remarks"]}'
        f"</div>",
        unsafe_allow_html=True,
    )

    st.write("")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Correct", result["correct"])

    with col2:
        st.metric("Wrong", result["wrong"])

    with col3:
        st.metric("Unanswered", result["unanswered"])

    with col4:
        st.metric(
            "Percentage",
            f'{result["percentage"]:.2f}%',
        )

    st.progress(result["percentage"] / 100)

    st.divider()

    # Download result.
    try:
        excel_data = create_excel_file(result)
        filename = (
            f"quiz_result_"
            f"{sanitize_filename(st.session_state.topic)}.xlsx"
        )

        st.download_button(
            label="📥 Download Result Sheet (Excel)",
            data=excel_data,
            file_name=filename,
            mime=(
                "application/vnd.openxmlformats-officedocument."
                "spreadsheetml.sheet"
            ),
            type="primary",
            use_container_width=True,
        )

    except Exception as error:
        st.error(
            "The result was calculated, but the Excel file "
            "could not be created."
        )
        st.caption(f"Technical detail: {error}")

    st.divider()

    st.subheader("📝 Answer Review")

    for index, question in enumerate(st.session_state.questions):
        user_answer = st.session_state.answers.get(index)
        correct_answer = question["correct_answer"]

        with st.expander(
            f"Question {index + 1}: {question['question']}"
        ):
            if user_answer is None:
                st.warning("Unanswered")
                st.write(
                    "**Correct Answer:** "
                    f"{question['options'][correct_answer]}"
                )

            elif user_answer == correct_answer:
                st.success(
                    "Correct — "
                    f"{question['options'][correct_answer]}"
                )

            else:
                st.error(
                    "Wrong — "
                    f"Your answer: {question['options'][user_answer]}"
                )
                st.info(
                    "Correct answer: "
                    f"{question['options'][correct_answer]}"
                )

    st.divider()

    if st.button(
        "🔄 Start New Quiz",
        type="primary",
        use_container_width=True,
    ):
        reset_quiz()
        st.rerun()


# ============================================================
# MAIN APP ROUTER
# ============================================================

if st.session_state.page == "start":
    show_start_page()

elif st.session_state.page == "quiz":
    show_quiz_page()

elif st.session_state.page == "result":
    show_result_page()

else:
    # Safety fallback if session state somehow becomes invalid.
    reset_quiz()
    st.rerun()
