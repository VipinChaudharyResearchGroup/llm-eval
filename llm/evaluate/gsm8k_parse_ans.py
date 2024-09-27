import re


def clean_response(response, final_answer_trigger="therefore"):
    response_text = " ".join(response.strip().split())

    parts = re.split(final_answer_trigger, response_text, flags=re.IGNORECASE)
    if len(parts) > 1:
        expected_text = parts[1].strip()
    else:
        parts = re.split("therefore", response_text, flags=re.IGNORECASE)
        expected_text = parts[1].strip() if len(parts) > 1 else response_text

    expected_text = expected_text.replace("$", "").replace(",", "")

    if "pm" in expected_text.lower() or "am" in expected_text.lower():
        time_match = re.search(
            r"(\d{1,2}:\d{2})\s*(am|pm)", expected_text, re.IGNORECASE
        )
        if time_match:
            time_str, meridiem = time_match.groups()
            hours, minutes = map(int, time_str.split(":"))

            if meridiem.lower() == "pm" and hours != 12:
                hours += 12
            elif meridiem.lower() == "am" and hours == 12:
                hours = 0

            time_float = hours + minutes / 60.0
            return round(time_float, 2)

    numbers = re.findall(r"-?\d+\.?\d*", expected_text)
    parsed_numbers = [float(num) if "." in num else int(num) for num in numbers]

    if parsed_numbers:
        return parsed_numbers[-1]
    else:
        exception_phrases = [
            "there is no",
            "there are no",
            "there will be no",
            "there will not be",
        ]
        if any(phrase in expected_text.lower() for phrase in exception_phrases):
            return 0

        response_before_question = response.split("Q:")[0]
        lines = response_before_question.strip().split("\n")
        for line in reversed(lines):
            line = line.strip()
            if line:
                parts = re.split(final_answer_trigger, line, flags=re.IGNORECASE)
                expected_text = parts[1].strip() if len(parts) > 1 else line
                expected_text = expected_text.replace("$", "").replace(",", "")
                numbers = re.findall(r"-?\d+\.?\d*", expected_text)
                parsed_numbers = [
                    float(num) if "." in num else int(num) for num in numbers
                ]
                if parsed_numbers:
                    return parsed_numbers[-1]

    return None


if __name__ == "__main__":
    final_answer_trigger = "The answer is"
    # final_answer_trigger = "Therefore, the final answer is"
    # edge cases have been removed since the repo is public
    test_cases = {
        "case1": {
            "response": "Something. Therefore, the final answer is -$1,000.00.",
            "expected": -1000.0,
        },
    }

    for case_name, case_data in test_cases.items():
        response = case_data["response"]
        expected = case_data["expected"]
        result = clean_response(response, final_answer_trigger)
        is_correct = result == expected
        print(f"{case_name}: {result} == {expected} --> {is_correct}")
