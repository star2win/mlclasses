from transformers import pipeline

classifier = pipeline("sentiment-analysis")
result = classifier("I like this class")

# Print in a more detailed format
for item in result:
    print(f"\n\nLabel: {item['label']}")
    print(f"Score: {item['score']:.4f}\n\n")

classifier2 = pipeline("zero-shot-classification")
result2 = classifier2(
    "This is a course about the Transofrmers library",
    candidate_labels=["education", "politics", "business"],
)

print(f"\n\nZero show classification: {result2}\n\n")

generator = pipeline("text-generation")
result3 = generator("In this course, we will taech yo how to")

print(f"\n\nText generation example: {result3}\n\n")

generator2 = pipeline("text-generation", model="distilgpt2")
result4 = generator2("In this course, we will taech yo how to",
                     max_length=30,
                     num_return_sequences=2,
                     )

print(f"\n\nText generation example: {result4}\n\n")

unmasker = pipeline("fill-mask")
result5 = unmasker("This course will teach you all about <mask> models.", top_k=2)

print(f"\n\nText generation example: {result5}\n\n")

ner = pipeline("ner", grouped_entities=True)
result6 = ner("My name is Sylvain and I work at Hugging Face in Brooklyn.")

print(f"\n\nText generation example: {result6}\n\n")

question_answerer = pipeline("question-answering")
result7 = question_answerer(
    question="Where do I work?",
    context="My name is Sylvain and I work at Hugging Face in Brooklyn",
)

print(f"\n\nText generation example: {result7}\n\n")

summarizer = pipeline("summarization")
result8 = summarizer(
        """
    America has changed dramatically during recent years. Not only has the number of 
    graduates in traditional engineering disciplines such as mechanical, civil, 
    electrical, chemical, and aeronautical engineering declined, but in most of 
    the premier American universities engineering curricula now concentrate on 
    and encourage largely the study of engineering science. As a result, there 
    are declining offerings in engineering subjects dealing with infrastructure, 
    the environment, and related issues, and greater concentration on high 
    technology subjects, largely supporting increasingly complex scientific 
    developments. While the latter is important, it should not be at the expense 
    of more traditional engineering.

    Rapidly developing economies such as China and India, as well as other 
    industrial countries in Europe and Asia, continue to encourage and advance 
    the teaching of engineering. Both China and India, respectively, graduate 
    six and eight times as many traditional engineers as does the United States. 
    Other industrial countries at minimum maintain their output, while America 
    suffers an increasingly serious decline in the number of engineering graduates 
    and a lack of well-educated engineers.
    """
)

print(f"\n\nText generation example: {result8}\n\n")

translator = pipeline("translation", model="Helsinki-NLP/opus-mt-pl-en")
result9 = translator("Kto to jest?")

print(f"\n\nText generation example: {result9}\n\n")
