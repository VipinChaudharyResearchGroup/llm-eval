import re


def clean_response(response, final_answer_trigger="therefore"):

    response_ = response.lower().strip().split("\n")[0]

    first_response = response_.split(final_answer_trigger.lower())

    if len(first_response) > 1:  # final answer trigger found
        expected = first_response[1].split("\n")[0].strip()[:-1]
    else:
        expected = response_.split("therefore")[-1]

    expected = expected.replace("$", "").replace(",", "")

    ## TIME
    if "pm" or "am" in expected:
        time_match = re.search(r"(\d{1,2}:\d{2} (?:am|pm))", expected, re.IGNORECASE)
        if time_match:
            time_str = time_match.group(1)

            time, _ = time_str.split()
            hours, minutes = map(int, time.split(":"))
            decimal_minutes = minutes / 60.0

            time_float = hours + decimal_minutes

            return round(time_float, 2)
    ## TIME

    numbers = re.findall(r"-?\d+\.?\d*", expected)

    parsed_numbers = [float(num) if "." in num else int(num) for num in numbers]

    num_parsed = len(parsed_numbers)

    if num_parsed > 0:
        return parsed_numbers[-1]
    else:
        exception_candidates = [
            "there is no",
            "there are no",
            "there will be no",
            "there will not be",
            "there will not be",
        ]

        for candidate in exception_candidates:
            if candidate in expected:
                return 0

        cle = response.split("Q:")[0].split("\n")
        l = len(cle)

        for i in range(l):
            if len(cle[l - i - 1]) > 0:
                x = cle[l - i - 1].lower().split(final_answer_trigger.lower())
                expected = x[-1].replace("$", "").replace(",", "")

                numbers = re.findall(r"-?\d+\.?\d*", expected)
                parsed_numbers = [
                    float(num) if "." in num else int(num) for num in numbers
                ]
                num_parsed = len(parsed_numbers)
                if num_parsed > 0:
                    return parsed_numbers[-1]

        return None


if __name__ is "__main__":
    final_answer_trigger = "The answer is"
    # final_answer_trigger = "Therefore, the final answer is"
    response = (
        "The cost of 1 pencil is $1.50. Therefore, the cost of 5 pencils is $7.50."
    )
    print(clean_response(response))  # 7.5
