# Evaluation & Testing Report
**AI Training Assistant for New Employees**

Generated: 2026-08-18 17:39 UTC
LLM backend: `extractive` · Retrieval backend: `tfidf` · Router backend: `rule_based`

---

## 1. Accuracy Metrics

| Metric | Score |
|---|---|
| Routing accuracy | 19/20 = 95.0% |
| Citation match rate (RAG routes) | 16/17 = 94.1% |
| Key-phrase presence rate | 13/17 = 76.5% |

Routing accuracy measures whether the query router selected the same route a human
labeler expected. Citation match rate measures whether the correct source document was
among the retrieved/cited chunks for document-grounded routes. Key-phrase presence rate
is a loose proxy for answer correctness/completeness (does the generated answer contain
at least one of the labeled key phrases).

## 2. Response Quality Analysis

- **Average answer length:** 550 characters (σ = 150)
- **Shortest / longest answer:** 248 / 746 characters
- **Answers with at least one citation:** 17/20 (85.0%)

**Breakdown by predicted route:**

| Route | Questions | Avg. answer length | With citation |
|---|---|---|---|
| `admin_policy` | 11 | 593 chars | 11/11 |
| `direct_llm` | 3 | 248 chars | 0/3 |
| `general_company` | 2 | 580 chars | 2/2 |
| `role_specific` | 4 | 644 chars | 4/4 |

## 3. Test Cases

All 20 labeled questions from `data/evaluation_set.csv`, run end-to-end through routing -> retrieval -> generation.

| ID | Question | Expected Route | Predicted Route | Route OK | Citation OK | Key-Phrase OK |
|---|---|---|---|---|---|---|
| Q01 | What are the company's core values and where should I document deci... | `general_company` | `general_company` | ✅ | ✅ | ✅ |
| Q02 | What are standard work hours and the core collaboration window? | `general_company` | `general_company` | ✅ | ✅ | ✅ |
| Q03 | What tools are considered common internal tools during onboarding? | `general_company` | `admin_policy` | ❌ | ❌ | ✅ |
| Q04 | As a Data Analyst, what are the first 30 days expectations? | `role_specific` | `role_specific` | ✅ | ✅ | ✅ |
| Q05 | Who should a Product Manager ask for feasibility checks and why? | `role_specific` | `role_specific` | ✅ | ✅ | ✅ |
| Q06 | What are key responsibilities of a Product Manager in this organiza... | `role_specific` | `role_specific` | ✅ | ✅ | ✅ |
| Q07 | Which tools are commonly used by a Data Analyst here? | `role_specific` | `role_specific` | ✅ | ✅ | ✅ |
| Q08 | How do I submit an expense claim and by when? | `admin_policy` | `admin_policy` | ✅ | ✅ | ✅ |
| Q09 | Are meal expenses reimbursable during business travel and what is t... | `admin_policy` | `admin_policy` | ✅ | ✅ | ❌ |
| Q10 | How do I request PTO and what notice is expected for planned leave? | `admin_policy` | `admin_policy` | ✅ | ✅ | ❌ |
| Q11 | What should I do if I can’t access a tool on Day 1? | `admin_policy` | `admin_policy` | ✅ | ✅ | ❌ |
| Q12 | What is the standard turnaround time for access requests? | `admin_policy` | `admin_policy` | ✅ | ✅ | ✅ |
| Q13 | Do I need to fill timesheets and when are they due? | `admin_policy` | `admin_policy` | ✅ | ✅ | ❌ |
| Q14 | What is the reimbursement timeline after an expense is approved? | `admin_policy` | `admin_policy` | ✅ | ✅ | ✅ |
| Q15 | What are the steps to take if I suspect a security incident? | `admin_policy` | `admin_policy` | ✅ | ✅ | ✅ |
| Q16 | How do I request an employment verification letter? | `admin_policy` | `admin_policy` | ✅ | ✅ | ✅ |
| Q17 | How many PTO days can be carried forward? | `admin_policy` | `admin_policy` | ✅ | ✅ | ✅ |
| Q18 | What is my exact salary breakup and tax deductions? | `direct_llm` | `direct_llm` | ✅ | — | — |
| Q19 | Can you approve my leave request right now? | `direct_llm` | `direct_llm` | ✅ | — | — |
| Q20 | Write a personal performance improvement plan for me based on my ma... | `direct_llm` | `direct_llm` | ✅ | — | — |

## 4. User Feedback Results

No user feedback has been recorded yet. Feedback accumulates automatically as real users click 👍 / 👎 on answers in the Streamlit app (`app.py`) - re-run this report after some usage to populate this section.

---

## 5. How to reproduce

```bash
python generate_report.py                       # default backends (tfidf / rule_based)
python generate_report.py --retrieval faiss      # FAISS vector-DB retrieval
python generate_report.py --router llm           # LLM-based query classification
```
