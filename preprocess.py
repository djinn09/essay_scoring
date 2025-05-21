def eval_based_on_length(answer: str, min_length: int = 8) -> bool:
    """
    Params:
        answer: str candidate answer
        min_length: int minim.um length of candidate answer should be
    Returns:
        bool
    function check if given string is less than of certain min_length

    """

    evaluate = False
    # if answer is not string
    if not isinstance(answer, str):
        return evaluate

    # split the string
    answer = answer.split(" ")
    if len(answer) > min_length:
        evaluate = True

    return evaluate


def remove_unbalanced_repetition(answer: str, top_words_p: float = 0.4, boundary: float = 0.3):
    splinted = answer.split(" ")
    doc_fre = len(splinted)

    c = Counter(splinted)

    top = math.ceil(doc_fre % top_words_p)

    count = 0
    values = list(c.values())
    values.sort(reverse=True)
    for i in range(top + 1):
        # to resolve out of bounds error
        if i < len(values):
            # print(values[i])
            count += values[i]

    fre = count / doc_fre

    return True if fre < boundary else False


def eval_based_on_repetition(answer: str, percentage: float = 0.6) -> bool:
    """
    Function checks if a keyword is repeated more than some percentage
    in answer, then that answer is improper
    Params:
        answer: str
        percentage: float
    Returns:
        bool
    """
    evaluate = False
    if not answer or not isinstance(answer, str):
        return evaluate

    # split the answer into tokens
    # split by space as speech recognition does not give
    # commas
    answer = answer.strip().lower()
    # split string by space and remove empty words
    keywords = [i for i in answer.split(" ") if i]
    total_words = len(keywords)
    keywords_dict = Counter(keywords)
    # use key later
    for _, value in keywords_dict.items():
        keyword_percentage = value / total_words
        if keyword_percentage > percentage:
            return evaluate

    return True


def remove_punch(text):
    return text.translate(table)


# checks if the candidate response contains the question
def que_check(answer: str, question: str):
    if answer == question:
        return False
    else:
        return True


# removesthe question text in the candidate response if present
def find_first_non_repeating_sentence(string: str):
    tokens = string.split()
    found = []
    for token in tokens:
        # look until token is not repeated
        if token not in found:
            found.append(token)
        else:
            break

    return " ".join(found)


# find if a sentence is repeating again and again
def if_repeating_sentence(answer: str):
    """
    go till non repeating characters in sentence and create sentence out
    of it and check it is repeating again and again
    """
    answer = answer.lower()
    answer = remove_punch(answer)
    n_r_sentence = find_first_non_repeating_sentence(answer)
    # a case could be to check if sentences are repeated one after another
    # fair case could be a sentence might be repeated else where.
    sentences = re.findall(n_r_sentence, answer)
    if len(sentences) > 1:
        return True
    return False


def remove_ques_text(answer: str, question: str):
    if question in answer:
        answer = answer.strip(question)

    return answer


def get_unique_words_from_string(sentence: str) -> set:
    if not isinstance(sentence, str):
        raise TypeError(f"Expected string but got: {type(sentence)}")

    sentence = sentence.translate(str.maketrans("", "", string.punctuation))
    sentence = sentence.lower()
    # stop_words = set(stopwords.words("english"))
    stop_words = stopwords
    word_tokens = word_tokenize(sentence)
    filtered_sentence = set([w for w in word_tokens if w not in stop_words])
    # output = set(filtered_sentence
    return filtered_sentence




def if_only_keywords(tags_dict: Dict[str, str]) -> bool:
    # must have tags
    # check sentence by sentence
    required_tags = {'AUX', 'DET'}
    # print(tags_dict)
    arr = []  # unique tags
    for i in tags_dict.values():
        if i not in arr:
            arr.append(i)

    # all tags should be present in sentence
    # if even a single tag is not present than make it only keywords
    print("tags:", arr)
    arr = set(arr)
    count = 0
    for tag in required_tags:
        if tag in arr:
            count += 1

    return count != len(required_tags)