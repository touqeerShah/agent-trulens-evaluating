"""The same five QA cases used by the original graph experiments."""

QUESTIONS = [
    {
        "name": "market_cap",
        "question": "Chart the current market capitalization of the top 5 banks in the US.",
        "context": "Bank A has market capitalization 100. Bank B has market capitalization 80. Bank C has market capitalization 60.",
        "reference": "Identify the largest US banks and present market-cap values with dates and sources.",
    },
    {
        "name": "regulatory_changes",
        "question": "Identify current regulatory changes for the financial services industry in the US.",
        "context": "A new reporting rule changes incident disclosure timelines. A capital rule proposal is open for public comment.",
        "reference": "Current US financial-services changes include incident reporting and capital-rule proposals.",
    },
    {
        "name": "top_client_deals",
        "question": "What are our top 3 client deals? Chart the deal value for each.",
        "context": "Atlas Health deal value is 120000 EUR. Northstar Labs deal value is 75000 EUR. BluePeak Pharmacy deal value is 45000 EUR.",
        "reference": "Atlas Health is first at 120000 EUR, Northstar Labs second at 75000 EUR, and BluePeak Pharmacy third at 45000 EUR.",
    },
    {
        "name": "pending_deals",
        "question": "Identify pending deals, connect them to regulatory changes, and propose a value proposition for each customer.",
        "context": "BluePeak Pharmacy is pending because of a compliance review. The new rule requires audit exports. BluePeak needs shorter incident response times.",
        "reference": "Recommend audit exports and shorter incident response as the value proposition for BluePeak Pharmacy.",
    },
    {
        "name": "largest_client_news",
        "question": "Identify our largest client deal, find important meeting topics, and find related news.",
        "context": "Atlas Health is our largest client deal at 120000 EUR. The Atlas meeting discussed audit exports and incident response. Recent industry news discusses stricter incident reporting.",
        "reference": "Atlas Health is largest; topics are audit exports and incident response; related news concerns stricter incident reporting.",
    },
]

# The exact three questions recorded in the original Lesson 4 example.
LESSON4_QUESTIONS = [QUESTIONS[2], QUESTIONS[3], QUESTIONS[4]]
