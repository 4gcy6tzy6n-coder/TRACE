"""Generate 150 additional reasoning samples (30 per type) for v0.3.

Run: python data/generate_extra_samples.py
Output: data/toy_reasoning_extended.jsonl (200 total)
"""

import json
from pathlib import Path

# ─── Templates ───

DIRECT_EVIDENCE = [
    # Objects, facts, direct lookup
    {"evidence": ["Doc A: The Great Pyramid was built around 2560 BC.", "Doc B: It is located in Giza, Egypt."], "question": "When was the Great Pyramid built?", "gold_answer": "around 2560 BC", "gold_evidence_span": "built around 2560 BC"},
    {"evidence": ["Doc A: The chemical symbol for gold is Au.", "Doc B: Gold has atomic number 79."], "question": "What is the chemical symbol for gold?", "gold_answer": "Au", "gold_evidence_span": "symbol for gold is Au"},
    {"evidence": ["Doc A: Tokyo is the capital of Japan.", "Doc B: Japan has a population of 125 million."], "question": "What is the capital of Japan?", "gold_answer": "Tokyo", "gold_evidence_span": "capital of Japan is Tokyo"},
    {"evidence": ["Doc A: The Moon orbits Earth at 3683 km/h.", "Doc B: The Moon is 384400 km from Earth."], "question": "How fast does the Moon orbit Earth?", "gold_answer": "3683 km/h", "gold_evidence_span": "3683 km/h"},
    {"evidence": ["Doc A: Beethoven was born in 1770.", "Doc B: He composed Symphony No. 5 in 1808."], "question": "When was Beethoven born?", "gold_answer": "1770", "gold_evidence_span": "born in 1770"},
    {"evidence": ["Doc A: The Pacific Ocean covers 165 million sq km.", "Doc B: It is the largest ocean on Earth."], "question": "How large is the Pacific Ocean?", "gold_answer": "165 million sq km", "gold_evidence_span": "165 million sq km"},
    {"evidence": ["Doc A: The human body has 206 bones.", "Doc B: Bones are connected by joints."], "question": "How many bones does the human body have?", "gold_answer": "206", "gold_evidence_span": "206 bones"},
    {"evidence": ["Doc A: Shakespeare wrote Hamlet in 1601.", "Doc B: Hamlet is set in Denmark."], "question": "When did Shakespeare write Hamlet?", "gold_answer": "1601", "gold_evidence_span": "wrote Hamlet in 1601"},
    {"evidence": ["Doc A: The Nile River is 6650 km long.", "Doc B: It flows through Egypt and Sudan."], "question": "How long is the Nile River?", "gold_answer": "6650 km", "gold_evidence_span": "6650 km long"},
    {"evidence": ["Doc A: Python was first released in 1991.", "Doc B: It was designed by Guido van Rossum."], "question": "When was Python first released?", "gold_answer": "1991", "gold_evidence_span": "released in 1991"},
    # Academic/scientific
    {"evidence": ["Doc A: DNA was discovered by Watson and Crick in 1953.", "Doc B: DNA carries genetic information."], "question": "Who discovered DNA?", "gold_answer": "Watson and Crick", "gold_evidence_span": "discovered by Watson and Crick"},
    {"evidence": ["Doc A: The first iPhone was released in 2007.", "Doc B: It was announced by Steve Jobs."], "question": "When was the first iPhone released?", "gold_answer": "2007", "gold_evidence_span": "released in 2007"},
    {"evidence": ["Doc A: The United Nations was founded in 1945.", "Doc B: It is headquartered in New York."], "question": "When was the United Nations founded?", "gold_answer": "1945", "gold_evidence_span": "founded in 1945"},
    {"evidence": ["Doc A: Mars has two moons named Phobos and Deimos.", "Doc B: Mars is the fourth planet from the Sun."], "question": "How many moons does Mars have?", "gold_answer": "two", "gold_evidence_span": "two moons"},
    {"evidence": ["Doc A: The Dead Sea is 430 meters below sea level.", "Doc B: It is one of the saltiest bodies of water."], "question": "How far below sea level is the Dead Sea?", "gold_answer": "430 meters", "gold_evidence_span": "430 meters below sea level"},
    # Organizations
    {"evidence": ["Doc A: Google was founded in 1998.", "Doc B: It was founded by Larry Page and Sergey Brin."], "question": "Who founded Google?", "gold_answer": "Larry Page and Sergey Brin", "gold_evidence_span": "founded by Larry Page and Sergey Brin"},
    {"evidence": ["Doc A: The FIFA World Cup is held every 4 years.", "Doc B: The first World Cup was held in 1930."], "question": "How often is the FIFA World Cup held?", "gold_answer": "every 4 years", "gold_evidence_span": "every 4 years"},
    {"evidence": ["Doc A: The Amazon rainforest covers 5.5 million sq km.", "Doc B: It spans nine countries."], "question": "How large is the Amazon rainforest?", "gold_answer": "5.5 million sq km", "gold_evidence_span": "5.5 million sq km"},
    {"evidence": ["Doc A: Albert Einstein published his theory of relativity in 1905.", "Doc B: He won the Nobel Prize in 1921."], "question": "When did Einstein publish his theory of relativity?", "gold_answer": "1905", "gold_evidence_span": "published his theory of relativity in 1905"},
    {"evidence": ["Doc A: The Titanic sank in 1912.", "Doc B: It hit an iceberg in the North Atlantic."], "question": "When did the Titanic sink?", "gold_answer": "1912", "gold_evidence_span": "sank in 1912"},
    # Geography
    {"evidence": ["Doc A: Canada has 10 provinces.", "Doc B: Canada's capital is Ottawa."], "question": "How many provinces does Canada have?", "gold_answer": "10", "gold_evidence_span": "10 provinces"},
    {"evidence": ["Doc A: The Sahara Desert covers 9.2 million sq km.", "Doc B: It is the largest hot desert."], "question": "How large is the Sahara Desert?", "gold_answer": "9.2 million sq km", "gold_evidence_span": "9.2 million sq km"},
    {"evidence": ["Doc A: Venus is the hottest planet in the solar system.", "Doc B: Its surface temperature reaches 462°C."], "question": "Which planet is the hottest?", "gold_answer": "Venus", "gold_evidence_span": "Venus is the hottest planet"},
    {"evidence": ["Doc A: The printing press was invented by Gutenberg in 1440.", "Doc B: It revolutionized book production."], "question": "Who invented the printing press?", "gold_answer": "Gutenberg", "gold_evidence_span": "invented by Gutenberg"},
    {"evidence": ["Doc A: Mount Kilimanjaro is 5895 meters high.", "Doc B: It is the highest mountain in Africa."], "question": "How high is Mount Kilimanjaro?", "gold_answer": "5895 meters", "gold_evidence_span": "5895 meters high"},
    {"evidence": ["Doc A: Facebook was launched in 2004.", "Doc B: It was founded by Mark Zuckerberg at Harvard."], "question": "When was Facebook launched?", "gold_answer": "2004", "gold_evidence_span": "launched in 2004"},
    {"evidence": ["Doc A: A blue whale can weigh up to 200 tons.", "Doc B: It is the largest animal on Earth."], "question": "How much can a blue whale weigh?", "gold_answer": "up to 200 tons", "gold_evidence_span": "weigh up to 200 tons"},
    {"evidence": ["Doc A: The first Olympic Games were held in 776 BC.", "Doc B: They took place in Olympia, Greece."], "question": "When were the first Olympic Games held?", "gold_answer": "776 BC", "gold_evidence_span": "held in 776 BC"},
    {"evidence": ["Doc A: Pluto was reclassified as a dwarf planet in 2006.", "Doc B: It was previously considered the ninth planet."], "question": "When was Pluto reclassified as a dwarf planet?", "gold_answer": "2006", "gold_evidence_span": "reclassified as a dwarf planet in 2006"},
    {"evidence": ["Doc A: The guitar typically has 6 strings.", "Doc B: Some guitars have 12 strings."], "question": "How many strings does a typical guitar have?", "gold_answer": "6", "gold_evidence_span": "6 strings"},
]

CONFLICT = [
    {"evidence": ["Doc A: Employee records show John was hired on March 15.", "Doc B: HR system shows John's start date was April 3."], "question": "When was John hired?", "gold_answer": "Cannot determine, records conflict", "gold_evidence_span": "March 15"},
    {"evidence": ["Doc A: Survey finds 60% of customers are satisfied.", "Doc B: Internal data shows only 35% renewal rate."], "question": "Are customers satisfied?", "gold_answer": "Cannot determine, satisfaction metrics conflict", "gold_evidence_span": "60% of customers are satisfied"},
    {"evidence": ["Doc A: Report says the project finished under budget.", "Doc B: Accounting shows the project was $2M over budget."], "question": "Was the project under budget?", "gold_answer": "Cannot determine, financial reports conflict", "gold_evidence_span": "under budget"},
    {"evidence": ["Doc A: The experiment produced a 15% improvement.", "Doc B: Replication showed only 2% improvement."], "question": "What is the true effect size?", "gold_answer": "Cannot determine, original and replication conflict", "gold_evidence_span": "15% improvement"},
    {"evidence": ["Doc A: Traffic camera shows the car ran a red light.", "Doc B: Witness says the light was still yellow."], "question": "Did the car run a red light?", "gold_answer": "Cannot determine, evidence conflicts", "gold_evidence_span": "ran a red light"},
    {"evidence": ["Doc A: Vendor claims the material is 95% pure.", "Doc B: Independent lab test shows 82% purity."], "question": "What is the purity of the material?", "gold_answer": "Cannot determine, tests conflict", "gold_evidence_span": "95% pure"},
    {"evidence": ["Doc A: Weather station recorded 45mm of rain.", "Doc B: Satellite data shows only 28mm of rain."], "question": "How much rain fell?", "gold_answer": "Cannot determine, measurements conflict", "gold_evidence_span": "45mm of rain"},
    {"evidence": ["Doc A: CEO states profit grew 20% this quarter.", "Doc B: SEC filing shows profit declined 5%."], "question": "Did profit grow this quarter?", "gold_answer": "Cannot determine, disclosed figures conflict", "gold_evidence_span": "grew 20%"},
    {"evidence": ["Doc A: Study links coffee to reduced dementia risk.", "Doc B: Meta-analysis finds no link between coffee and dementia."], "question": "Does coffee reduce dementia risk?", "gold_answer": "Evidence is conflicting", "gold_evidence_span": "reduced dementia risk"},
    {"evidence": ["Doc A: Product manual says battery lasts 10 hours.", "Doc B: Consumer tests show average battery life of 6 hours."], "question": "How long does the battery last?", "gold_answer": "Cannot determine, manufacturer and test data conflict", "gold_evidence_span": "10 hours"},
    {"evidence": ["Doc A: Invoice shows payment of $5,000 was received.", "Doc B: Bank statement shows no $5,000 deposit."], "question": "Was the $5,000 payment received?", "gold_answer": "Cannot determine, records conflict", "gold_evidence_span": "payment of $5,000"},
    {"evidence": ["Doc A: Inspection report rates the building as safe.", "Doc B: Engineer's assessment flags structural concerns."], "question": "Is the building safe?", "gold_answer": "Cannot determine, assessments conflict", "gold_evidence_span": "safe"},
    {"evidence": ["Doc A: Restaurant claims all ingredients are organic.", "Doc B: Supplier invoices show non-organic items."], "question": "Are all ingredients organic?", "gold_answer": "Cannot determine, claims and records conflict", "gold_evidence_span": "organic"},
    {"evidence": ["Doc A: Journal article reports treatment success rate of 90%.", "Doc B: Follow-up study reports 60% success rate."], "question": "What is the treatment success rate?", "gold_answer": "Cannot determine, studies conflict", "gold_evidence_span": "90%"},
    {"evidence": ["Doc A: Police report states speed was 45 mph at impact.", "Doc B: Accident reconstruction estimates 65 mph."], "question": "What was the speed at impact?", "gold_answer": "Cannot determine, estimates conflict", "gold_evidence_span": "45 mph"},
    {"evidence": ["Doc A: Software benchmark shows 30% faster performance.", "Doc B: User benchmarking shows no significant difference."], "question": "How much faster is the software?", "gold_answer": "Cannot determine, benchmarks conflict", "gold_evidence_span": "30% faster"},
    {"evidence": ["Doc A: Census data shows population of 1.2 million.", "Doc B: Utility records suggest 1.5 million residents."], "question": "What is the city's population?", "gold_answer": "Cannot determine, data sources conflict", "gold_evidence_span": "1.2 million"},
    {"evidence": ["Doc A: Audit found full compliance with regulations.", "Doc B: Whistleblower report documents multiple violations."], "question": "Is the company compliant?", "gold_answer": "Cannot determine, compliance reports conflict", "gold_evidence_span": "full compliance"},
    {"evidence": ["Doc A: Market research says 70% prefer brand X.", "Doc B: Sales data shows brand Y outsells brand X 3:1."], "question": "Which brand is preferred?", "gold_answer": "Cannot determine, survey and sales data conflict", "gold_evidence_span": "70% prefer brand X"},
    {"evidence": ["Doc A: The telescope detected water on the exoplanet.", "Doc B: Follow-up observations found no water signature."], "question": "Is there water on the exoplanet?", "gold_answer": "Cannot determine, observations conflict", "gold_evidence_span": "water"},
    {"evidence": ["Doc A: Real estate agent says house is 2500 sq ft.", "Doc B: Property tax records list house as 2200 sq ft."], "question": "How large is the house?", "gold_answer": "Cannot determine, measurements conflict", "gold_evidence_span": "2500 sq ft"},
    {"evidence": ["Doc A: Employee claims 40 hours worked this week.", "Doc B: Time tracking system shows 32 hours logged."], "question": "How many hours were worked?", "gold_answer": "Cannot determine, time records conflict", "gold_evidence_span": "40 hours"},
    {"evidence": ["Doc A: Manufacturer claims device is waterproof to 50m.", "Doc B: Test shows water damage at 30m depth."], "question": "Is the device waterproof to 50m?", "gold_answer": "Cannot determine, claims and test results conflict", "gold_evidence_span": "waterproof to 50m"},
    {"evidence": ["Doc A: Press release says app has 10 million users.", "Doc B: App store data shows 6 million downloads."], "question": "How many users does the app have?", "gold_answer": "Cannot determine, user numbers conflict", "gold_evidence_span": "10 million users"},
    {"evidence": ["Doc A: Doctor's note says patient can return to work.", "Doc B: Physical therapy assessment recommends 2 more weeks."], "question": "Can the patient return to work?", "gold_answer": "Cannot determine, medical opinions conflict", "gold_evidence_span": "return to work"},
    {"evidence": ["Doc A: Climate model predicts 2°C warming by 2050.", "Doc B: Alternative model predicts 4°C warming."], "question": "How much warming is predicted by 2050?", "gold_answer": "Cannot determine, models disagree", "gold_evidence_span": "2°C warming"},
    {"evidence": ["Doc A: University claims 95% graduation rate.", "Doc B: Government data shows 72% graduation rate."], "question": "What is the graduation rate?", "gold_answer": "Cannot determine, reported rates conflict", "gold_evidence_span": "95% graduation rate"},
    {"evidence": ["Doc A: Nutrition label says 150 calories per serving.", "Doc B: Independent lab finds 190 calories per serving."], "question": "How many calories per serving?", "gold_answer": "Cannot determine, nutritional data conflicts", "gold_evidence_span": "150 calories"},
    {"evidence": ["Doc A: Insurance adjuster estimates $5,000 in damages.", "Doc B: Repair shop quotes $8,500 for the same repairs."], "question": "What is the cost of repairs?", "gold_answer": "Cannot determine, estimates conflict", "gold_evidence_span": "$5,000"},
    {"evidence": ["Doc A: Biopsy results were negative for malignancy.", "Doc B: Second pathology review found malignant cells."], "question": "Is the tumor malignant?", "gold_answer": "Cannot determine, pathology reports conflict", "gold_evidence_span": "negative"},
]

EVIDENCE_GAP = [
    {"evidence": ["Doc A: The conference was held in Barcelona.", "Doc B: 500 people attended the conference."], "question": "What was the budget of the conference?", "gold_answer": "Cannot determine, budget not mentioned", "gold_evidence_span": "Barcelona"},
    {"evidence": ["Doc A: The new drug was approved by the FDA.", "Doc B: Clinical trials took 3 years."], "question": "What is the price of the drug?", "gold_answer": "Cannot determine, price not provided", "gold_evidence_span": "approved by the FDA"},
    {"evidence": ["Doc A: The author published her first book in 2015.", "Doc B: The book won a national award."], "question": "How many copies did the book sell?", "gold_answer": "Cannot determine, sales not mentioned", "gold_evidence_span": "published her first book"},
    {"evidence": ["Doc A: The company has offices in 12 countries.", "Doc B: The CEO was appointed in 2019."], "question": "What is the company's annual revenue?", "gold_answer": "Cannot determine, revenue not stated", "gold_evidence_span": "12 countries"},
    {"evidence": ["Doc A: The athlete won gold at the 2024 Olympics.", "Doc B: She trains 6 hours per day."], "question": "What is her world ranking?", "gold_answer": "Cannot determine, ranking not provided", "gold_evidence_span": "gold at the 2024 Olympics"},
    {"evidence": ["Doc A: The restaurant received a Michelin star.", "Doc B: The head chef trained in France."], "question": "How many seats does the restaurant have?", "gold_answer": "Cannot determine, capacity not mentioned", "gold_evidence_span": "Michelin star"},
    {"evidence": ["Doc A: The new processor has 16 cores.", "Doc B: It is manufactured using 3nm process."], "question": "What is the clock speed of the processor?", "gold_answer": "Cannot determine, clock speed not specified", "gold_evidence_span": "16 cores"},
    {"evidence": ["Doc A: The bridge was completed in 2018.", "Doc B: It connects two major cities."], "question": "How much did the bridge cost to build?", "gold_answer": "Cannot determine, cost not provided", "gold_evidence_span": "completed in 2018"},
    {"evidence": ["Doc A: The startup has 50 employees.", "Doc B: It is based in San Francisco."], "question": "What is the valuation of the startup?", "gold_answer": "Cannot determine, valuation not stated", "gold_evidence_span": "50 employees"},
    {"evidence": ["Doc A: The movie was directed by a first-time director.", "Doc B: It grossed $200M worldwide."], "question": "What was the production budget?", "gold_answer": "Cannot determine, budget not mentioned", "gold_evidence_span": "$200M worldwide"},
    # More diverse gaps
    {"evidence": ["Doc A: The professor teaches at MIT.", "Doc B: He has published 200 papers."], "question": "How many students does he teach per semester?", "gold_answer": "Cannot determine, class size not provided", "gold_evidence_span": "MIT"},
    {"evidence": ["Doc A: The hotel is rated 5 stars.", "Doc B: It has a rooftop pool."], "question": "How much does a night cost?", "gold_answer": "Cannot determine, price not listed", "gold_evidence_span": "5 stars"},
    {"evidence": ["Doc A: The spacecraft reached Mars orbit.", "Doc B: The mission launched in 2022."], "question": "How many astronauts were on board?", "gold_answer": "Cannot determine, crew size not stated", "gold_evidence_span": "Mars orbit"},
    {"evidence": ["Doc A: The novel was translated into 30 languages.", "Doc B: The author wrote it in 18 months."], "question": "How many pages is the novel?", "gold_answer": "Cannot determine, page count not mentioned", "gold_evidence_span": "30 languages"},
    {"evidence": ["Doc A: The team won the championship.", "Doc B: The coach has been with the team for 5 years."], "question": "What was the final score?", "gold_answer": "Cannot determine, score not provided", "gold_evidence_span": "won the championship"},
    {"evidence": ["Doc A: The painting was authenticated by experts.", "Doc B: It was created in the Renaissance period."], "question": "How much is the painting worth?", "gold_answer": "Cannot determine, value not stated", "gold_evidence_span": "authenticated by experts"},
    {"evidence": ["Doc A: The new phone has 5G connectivity.", "Doc B: It runs on the latest OS version."], "question": "What is the screen refresh rate?", "gold_answer": "Cannot determine, refresh rate not specified", "gold_evidence_span": "5G connectivity"},
    {"evidence": ["Doc A: The hurricane made landfall in Florida.", "Doc B: Wind speeds exceeded 150 mph."], "question": "How many people were evacuated?", "gold_answer": "Cannot determine, evacuation numbers not provided", "gold_evidence_span": "150 mph"},
    {"evidence": ["Doc A: The actress won an Oscar for her role.", "Doc B: The film was shot in 6 countries."], "question": "How much was she paid for the role?", "gold_answer": "Cannot determine, salary not disclosed", "gold_evidence_span": "Oscar"},
    {"evidence": ["Doc A: The airline operates 200 routes.", "Doc B: The fleet consists of Boeing aircraft."], "question": "How many passengers fly annually?", "gold_answer": "Cannot determine, passenger numbers not given", "gold_evidence_span": "200 routes"},
    {"evidence": ["Doc A: The software supports 15 languages.", "Doc B: Version 3.0 was released last month."], "question": "How many developers worked on version 3.0?", "gold_answer": "Cannot determine, team size not mentioned", "gold_evidence_span": "15 languages"},
    {"evidence": ["Doc A: The museum has 50,000 artifacts.", "Doc B: It opened in 1920."], "question": "How many visitors does it get annually?", "gold_answer": "Cannot determine, visitor count not provided", "gold_evidence_span": "50,000 artifacts"},
    {"evidence": ["Doc A: The manager has 15 years of experience.", "Doc B: He previously worked at two Fortune 500 companies."], "question": "What is his salary?", "gold_answer": "Cannot determine, salary not disclosed", "gold_evidence_span": "15 years"},
    {"evidence": ["Doc A: The diamond weighs 2 carats.", "Doc B: It has VS1 clarity."], "question": "What is the color grade of the diamond?", "gold_answer": "Cannot determine, color not specified", "gold_evidence_span": "2 carats"},
    {"evidence": ["Doc A: The hospital has 500 beds.", "Doc B: It employs 2,000 staff."], "question": "What is the patient satisfaction score?", "gold_answer": "Cannot determine, satisfaction data not provided", "gold_evidence_span": "500 beds"},
    {"evidence": ["Doc A: The podcast has 100 episodes.", "Doc B: The host is a former journalist."], "question": "How many weekly listeners does it have?", "gold_answer": "Cannot determine, listener count not provided", "gold_evidence_span": "100 episodes"},
    {"evidence": ["Doc A: The battery pack has 10,000 mAh capacity.", "Doc B: It supports fast charging."], "question": "How many charge cycles does the battery last?", "gold_answer": "Cannot determine, cycle life not specified", "gold_evidence_span": "10,000 mAh"},
    {"evidence": ["Doc A: The seminar was held virtually.", "Doc B: 300 people registered."], "question": "How long was the seminar?", "gold_answer": "Cannot determine, duration not mentioned", "gold_evidence_span": "virtually"},
    {"evidence": ["Doc A: The artist has released 5 albums.", "Doc B: His latest album topped the charts."], "question": "How many copies has his latest album sold?", "gold_answer": "Cannot determine, sales figures not disclosed", "gold_evidence_span": "5 albums"},
    {"evidence": ["Doc A: The recipe uses locally sourced ingredients.", "Doc B: Preparation takes 45 minutes."], "question": "How many calories per serving?", "gold_answer": "Cannot determine, nutritional info not provided", "gold_evidence_span": "locally sourced"},
]

MISLEADING_HINT = [
    {"evidence": ["Doc A: Celebrity endorsements claim the cream reverses aging.", "Doc B: Dermatologists say it is just a moisturizer."], "question": "Does the cream reverse aging?", "gold_answer": "No, dermatologists say it is just a moisturizer", "gold_evidence_span": "just a moisturizer"},
    {"evidence": ["Doc A: The sales team claims revenue grew significantly.", "Doc B: The audited financials show a 2% decline."], "question": "Did revenue grow?", "gold_answer": "No, audited financials show a 2% decline", "gold_evidence_span": "2% decline"},
    {"evidence": ["Doc A: The politician says crime is down.", "Doc B: Police statistics show crime rose 8% this year."], "question": "Is crime down?", "gold_answer": "No, police statistics show crime rose 8%", "gold_evidence_span": "rose 8%"},
    {"evidence": ["Doc A: Ads say the diet pill melts fat without exercise.", "Doc B: Clinical studies show no weight loss over placebo."], "question": "Does the diet pill cause weight loss?", "gold_answer": "No, clinical studies show no effect over placebo", "gold_evidence_span": "no weight loss over placebo"},
    {"evidence": ["Doc A: The car ad shows 50 MPG highway mileage.", "Doc B: EPA combined rating is 32 MPG."], "question": "What is the car's real fuel economy?", "gold_answer": "32 MPG combined per EPA", "gold_evidence_span": "32 MPG"},
    {"evidence": ["Doc A: Investor pitch claims the company is profitable.", "Doc B: SEC filings show operating losses for 3 years."], "question": "Is the company profitable?", "gold_answer": "No, SEC filings show operating losses", "gold_evidence_span": "operating losses"},
    {"evidence": ["Doc A: The website claims the hotel has ocean views.", "Doc B: Recent guests say rooms face a construction site."], "question": "Do rooms have ocean views?", "gold_answer": "No, recent guests say rooms face construction", "gold_evidence_span": "construction site"},
    {"evidence": ["Doc A: The company says it never shares user data.", "Doc B: Privacy policy allows sharing with affiliates."], "question": "Does the company share user data?", "gold_answer": "Yes, privacy policy allows sharing with affiliates", "gold_evidence_span": "allows sharing with affiliates"},
    {"evidence": ["Doc A: Brochure says the university has a 98% placement rate.", "Doc B: Alumni survey shows 65% employed in their field."], "question": "What is the real placement rate?", "gold_answer": "65% in their field per alumni survey", "gold_evidence_span": "65% employed in their field"},
    {"evidence": ["Doc A: The label says the juice is 100% natural.", "Doc B: Ingredient list includes artificial sweetener."], "question": "Is the juice 100% natural?", "gold_answer": "No, it contains artificial sweetener", "gold_evidence_span": "artificial sweetener"},
    {"evidence": ["Doc A: The CEO says no layoffs are planned.", "Doc B: Internal memo shows 500 positions being eliminated."], "question": "Are layoffs planned?", "gold_answer": "Yes, internal memo shows 500 positions being cut", "gold_evidence_span": "500 positions being eliminated"},
    {"evidence": ["Doc A: Ads claim the vacuum has 'no loss of suction'.", "Doc B: Consumer Reports found 40% suction loss after 6 months."], "question": "Does the vacuum maintain suction?", "gold_answer": "No, Consumer Reports found significant suction loss", "gold_evidence_span": "40% suction loss"},
    {"evidence": ["Doc A: The website says 'free shipping on all orders'.", "Doc B: Fine print shows minimum $50 purchase required."], "question": "Is shipping free on all orders?", "gold_answer": "No, minimum $50 purchase is required", "gold_evidence_span": "minimum $50 purchase"},
    {"evidence": ["Doc A: Influencers claim the headphones are noise-canceling.", "Doc B: Technical specs show only passive isolation."], "question": "Are the headphones noise-canceling?", "gold_answer": "No, they only have passive noise isolation", "gold_evidence_span": "passive isolation"},
    {"evidence": ["Doc A: Marketing says the app is 'used by millions'.", "Doc B: App store shows 50,000 total downloads."], "question": "How many users does the app have?", "gold_answer": "50,000 downloads per app store data", "gold_evidence_span": "50,000 total downloads"},
    {"evidence": ["Doc A: The realtor says the neighborhood is 'up and coming'.", "Doc B: Crime statistics show above-average incidents."], "question": "Is the neighborhood safe?", "gold_answer": "No, crime statistics show above-average incidents", "gold_evidence_span": "above-average incidents"},
    {"evidence": ["Doc A: The company says it is carbon neutral.", "Doc B: Environmental audit found unreported emissions."], "question": "Is the company truly carbon neutral?", "gold_answer": "No, audit found unreported emissions", "gold_evidence_span": "unreported emissions"},
    {"evidence": ["Doc A: Ads say the mattress lasts 20 years.", "Doc B: Warranty only covers 5 years of normal use."], "question": "How long does the mattress actually last under warranty?", "gold_answer": "5 years per warranty terms", "gold_evidence_span": "5 years"},
    {"evidence": ["Doc A: The brand claims all materials are sustainable.", "Doc B: Supply chain audit found non-certified suppliers."], "question": "Are all materials sustainable?", "gold_answer": "No, audit found non-certified suppliers", "gold_evidence_span": "non-certified suppliers"},
    {"evidence": ["Doc A: The startup claims 50% month-over-month growth.", "Doc B: Revenue data shows 5% actual growth."], "question": "What is the real growth rate?", "gold_answer": "5% per revenue data", "gold_evidence_span": "5% actual growth"},
    {"evidence": ["Doc A: The product claims to be 'dermatologist tested'.", "Doc B: The testing was done on 12 people."], "question": "Is the testing rigorous?", "gold_answer": "No, only 12 people were tested", "gold_evidence_span": "12 people"},
    {"evidence": ["Doc A: The ad says the course guarantees a job.", "Doc B: Only 30% of graduates found employment within 6 months."], "question": "Does the course guarantee a job?", "gold_answer": "No, only 30% found employment", "gold_evidence_span": "30% of graduates"},
    {"evidence": ["Doc A: The label says 'made with real fruit'.", "Doc B: Ingredients show fruit concentrate as 2% of content."], "question": "Is the product mainly made of fruit?", "gold_answer": "No, fruit concentrate is only 2%", "gold_evidence_span": "2%"},
    {"evidence": ["Doc A: The coach says the team is 'in great shape'.", "Doc B: Medical staff reports 5 injured starters."], "question": "Is the team healthy?", "gold_answer": "No, 5 starters are injured", "gold_evidence_span": "5 injured starters"},
    {"evidence": ["Doc A: The company says it values work-life balance.", "Doc B: Employee surveys show 60-hour average work weeks."], "question": "Does the company value work-life balance?", "gold_answer": "No, employees average 60-hour weeks", "gold_evidence_span": "60-hour average"},
    {"evidence": ["Doc A: The product says 'results in 7 days'.", "Doc B: Study shows results take 4-6 weeks on average."], "question": "How fast does the product work?", "gold_answer": "4-6 weeks per study data", "gold_evidence_span": "4-6 weeks"},
    {"evidence": ["Doc A: The restaurant claims to use 'fresh ingredients'.", "Doc B: Health inspection found frozen pre-made meals."], "question": "Does the restaurant use fresh ingredients?", "gold_answer": "No, inspection found frozen pre-made meals", "gold_evidence_span": "frozen pre-made"},
    {"evidence": ["Doc A: The company blog says they are 'hiring aggressively'.", "Doc B: LinkedIn shows headcount decreased by 10%."], "question": "Is the company growing its workforce?", "gold_answer": "No, headcount decreased", "gold_evidence_span": "decreased by 10%"},
    {"evidence": ["Doc A: The website shows 4.9-star average rating.", "Doc B: Independent review analysis suggests review manipulation."], "question": "Is the 4.9 rating trustworthy?", "gold_answer": "No, independent analysis suggests manipulation", "gold_evidence_span": "review manipulation"},
    {"evidence": ["Doc A: The supplement claims to 'boost immunity by 300%'.", "Doc B: Scientific consensus: no supplement boosts immunity by that amount."], "question": "Does the supplement boost immunity by 300%?", "gold_answer": "No, no supplement can boost immunity by that amount", "gold_evidence_span": "no supplement boosts immunity"},
]

MULTI_STEP = [
    {"evidence": ["Doc A: A factory produces 200 units per hour.", "Doc B: The factory operates 16 hours per day.", "Doc C: There is a 30-minute maintenance break each shift."], "question": "How many units are produced per day?", "gold_answer": "3200 units", "gold_evidence_span": "200 units per hour"},
    {"evidence": ["Doc A: Train A travels at 80 km/h.", "Doc B: Train B travels at 60 km/h.", "Doc C: They are 280 km apart heading toward each other."], "question": "How long until the trains meet?", "gold_answer": "2 hours", "gold_evidence_span": "280 km apart"},
    {"evidence": ["Doc A: A laptop costs $800 with a 15% discount.", "Doc B: There is also a $50 mail-in rebate.", "Doc C: Sales tax is 8%."], "question": "What is the final cost after all discounts and tax?", "gold_answer": "$616.40", "gold_evidence_span": "15% discount"},
    {"evidence": ["Doc A: A water tank holds 500 liters.", "Doc B: It fills at 25 liters per minute.", "Doc C: It currently has 150 liters."], "question": "How many minutes to fill the tank?", "gold_answer": "14 minutes", "gold_evidence_span": "25 liters per minute"},
    {"evidence": ["Doc A: The temperature at sunrise was 12°C.", "Doc B: It rises 3°C per hour.", "Doc C: At 2 PM the rate slows to 1°C per hour."], "question": "What is the temperature at 4 PM if sunrise was at 6 AM?", "gold_answer": "38°C", "gold_evidence_span": "rises 3°C per hour"},
    {"evidence": ["Doc A: A pizza has 8 slices.", "Doc B: 3 people each eat 2 slices.", "Doc C: 1 person eats 1 slice."], "question": "How many slices are left?", "gold_answer": "1 slice", "gold_evidence_span": "8 slices"},
    {"evidence": ["Doc A: The classroom has 5 rows of desks.", "Doc B: Each row has 6 desks.", "Doc C: 4 desks are empty."], "question": "How many students are in the classroom?", "gold_answer": "26 students", "gold_evidence_span": "5 rows"},
    {"evidence": ["Doc A: A book costs $24.", "Doc B: A bookmark costs $3.", "Doc C: You buy 3 books and 2 bookmarks with a 10% member discount."], "question": "What is the total cost?", "gold_answer": "$70.20", "gold_evidence_span": "$24"},
    {"evidence": ["Doc A: A movie started at 7:15 PM.", "Doc B: It runs for 2 hours 25 minutes.", "Doc C: There are 15 minutes of previews before the movie."], "question": "What time does the movie end?", "gold_answer": "9:55 PM", "gold_evidence_span": "2 hours 25 minutes"},
    {"evidence": ["Doc A: The perimeter of a square is 40 cm.", "Doc B: You need to find the area.", "Doc C: Area = side length squared."], "question": "What is the area of the square?", "gold_answer": "100 square cm", "gold_evidence_span": "40 cm"},
    {"evidence": ["Doc A: A restaurant bill is $120.", "Doc B: The group wants to leave a 20% tip.", "Doc C: The bill is split 4 ways."], "question": "How much does each person pay?", "gold_answer": "$36", "gold_evidence_span": "20% tip"},
    {"evidence": ["Doc A: An investor puts $5,000 in a fund.", "Doc B: The fund grows 7% per year.", "Doc C: After 3 years, she withdraws half."], "question": "How much does she withdraw?", "gold_answer": "$3,062.50", "gold_evidence_span": "7% per year"},
    {"evidence": ["Doc A: A concert venue seats 1,200 people.", "Doc B: 3/4 of tickets are sold in advance at $50.", "Doc C: Remaining tickets sell at $65 at the door."], "question": "What is the total revenue if all tickets sell?", "gold_answer": "$64,500", "gold_evidence_span": "1,200 people"},
    {"evidence": ["Doc A: A recipe needs 3 cups of flour for 24 cookies.", "Doc B: You want to make 60 cookies.", "Doc C: You already have 2 cups of flour."], "question": "How much more flour do you need?", "gold_answer": "5.5 more cups", "gold_evidence_span": "3 cups for 24 cookies"},
    {"evidence": ["Doc A: A tank holds 60 gallons.", "Doc B: Pipe A fills at 4 gallons/minute.", "Doc C: Pipe B drains at 1 gallon/minute."], "question": "How many minutes to fill the empty tank with both pipes open?", "gold_answer": "20 minutes", "gold_evidence_span": "4 gallons/minute"},
    {"evidence": ["Doc A: A photographer takes 100 photos per hour.", "Doc B: She works 6 hours.", "Doc C: 15% of photos are deleted as unusable."], "question": "How many usable photos does she have?", "gold_answer": "510 photos", "gold_evidence_span": "100 photos per hour"},
    {"evidence": ["Doc A: The original price of a jacket is $180.", "Doc B: It is marked down 30%.", "Doc C: An additional 15% is taken off at checkout."], "question": "What is the final price?", "gold_answer": "$107.10", "gold_evidence_span": "30%"},
    {"evidence": ["Doc A: A runner covers 5 km in 20 minutes.", "Doc B: She maintains the same pace for a full marathon.", "Doc C: Marathon distance is 42.2 km."], "question": "How long will the marathon take at this pace?", "gold_answer": "2 hours 49 minutes", "gold_evidence_span": "5 km in 20 minutes"},
    {"evidence": ["Doc A: The sum of two numbers is 50.", "Doc B: The difference is 10.", "Doc C: The larger number is the answer."], "question": "What is the larger number?", "gold_answer": "30", "gold_evidence_span": "sum of two numbers is 50"},
    {"evidence": ["Doc A: A cylinder has radius 5 cm and height 10 cm.", "Doc B: Volume = pi * r^2 * h.", "Doc C: Use pi = 3.14."], "question": "What is the volume of the cylinder?", "gold_answer": "785 cubic cm", "gold_evidence_span": "radius 5 cm"},
    {"evidence": ["Doc A: A store has 240 items in stock.", "Doc B: They sell 1/6 in the morning.", "Doc C: They sell 1/4 of the remainder in the afternoon."], "question": "How many items are left?", "gold_answer": "150 items", "gold_evidence_span": "240 items"},
    {"evidence": ["Doc A: A loan of $10,000 at 5% simple interest.", "Doc B: Interest accrues over 3 years.", "Doc C: Monthly payments are $300."], "question": "What is the total interest paid?", "gold_answer": "$1,500", "gold_evidence_span": "5% simple interest"},
    {"evidence": ["Doc A: A bus carries 45 passengers per trip.", "Doc B: It makes 8 trips per day.", "Doc C: The bus runs 5 days per week."], "question": "How many passengers per week?", "gold_answer": "1800 passengers", "gold_evidence_span": "45 passengers per trip"},
    {"evidence": ["Doc A: A garden is 12m by 8m.", "Doc B: A path 1m wide is built around it.", "Doc C: You need to find the total area including the path."], "question": "What is the total area including the path?", "gold_answer": "140 square meters", "gold_evidence_span": "12m by 8m"},
    {"evidence": ["Doc A: A computer processes 50 tasks per second.", "Doc B: Each task takes 0.02 seconds.", "Doc C: There are 1000 tasks in the queue."], "question": "How many seconds to process all tasks?", "gold_answer": "20 seconds", "gold_evidence_span": "50 tasks per second"},
    {"evidence": ["Doc A: The ratio of cats to dogs is 3:5.", "Doc B: There are 40 total pets.", "Doc C: You need to find the number of cats."], "question": "How many cats are there?", "gold_answer": "15 cats", "gold_evidence_span": "ratio of cats to dogs is 3:5"},
    {"evidence": ["Doc A: A jar contains red and blue marbles.", "Doc B: 60% are red, and there are 24 blue marbles.", "Doc C: You need the total count."], "question": "How many marbles in total?", "gold_answer": "60 marbles", "gold_evidence_span": "60% are red"},
    {"evidence": ["Doc A: A flight covers 3000 miles at 500 mph.", "Doc B: There is a 45-minute layover.", "Doc C: Boarding takes 30 minutes before departure."], "question": "What is the total travel time?", "gold_answer": "7 hours 15 minutes", "gold_evidence_span": "500 mph"},
    {"evidence": ["Doc A: A recipe uses flour and sugar in a 3:1 ratio.", "Doc B: You use 4 cups of sugar.", "Doc C: Flour is the target quantity."], "question": "How much flour do you need?", "gold_answer": "12 cups", "gold_evidence_span": "3:1 ratio"},
    {"evidence": ["Doc A: The area of a triangle is 60 square inches.", "Doc B: The base is 12 inches.", "Doc C: Area = 1/2 * base * height."], "question": "What is the height of the triangle?", "gold_answer": "10 inches", "gold_evidence_span": "60 square inches"},
]

# ─── Build extended dataset ───

def generate_extended_dataset():
    base_path = Path(__file__).parent
    existing_path = base_path / "toy_reasoning.jsonl"

    # Load existing samples
    existing = []
    with open(existing_path) as f:
        for line in f:
            line = line.strip()
            if line:
                existing.append(json.loads(line))

    next_id = len(existing) + 1
    new_samples = []

    def add_samples(templates, reasoning_type, max_count=30):
        nonlocal next_id
        count = 0
        for tmpl in templates:
            if count >= max_count:
                break
            sid = f"case_{next_id:03d}"
            sample = {
                "id": sid,
                "evidence": tmpl["evidence"],
                "question": tmpl["question"],
                "gold_answer": tmpl["gold_answer"],
                "gold_evidence_span": tmpl["gold_evidence_span"],
                "reasoning_type": reasoning_type,
                "gold_thought_steps": [
                    f"Identify the target information.",
                    f"Find the relevant evidence.",
                    f"Extract the answer: {tmpl['gold_answer']}.",
                ],
                "label": "faithful",
            }
            new_samples.append(sample)
            next_id += 1
            count += 1

    add_samples(DIRECT_EVIDENCE, "direct_evidence", 30)
    add_samples(CONFLICT, "conflict", 30)
    add_samples(EVIDENCE_GAP, "evidence_gap", 30)
    add_samples(MISLEADING_HINT, "misleading_hint", 30)
    add_samples(MULTI_STEP, "multi_step", 30)

    # Combine and save
    all_samples = existing + new_samples
    output_path = base_path / "toy_reasoning.jsonl"
    with open(output_path, "w") as f:
        for s in all_samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    # Count by type
    from collections import Counter
    counts = Counter(s["reasoning_type"] for s in all_samples)
    print(f"Total samples: {len(all_samples)}")
    for rt, cnt in sorted(counts.items()):
        print(f"  {rt}: {cnt}")
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    generate_extended_dataset()
