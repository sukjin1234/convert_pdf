# RAG acceptance report

- Generated: 2026-08-23T04:06:55.713284+00:00
- Dataset: `0fd367d8-5003-440a-92bc-e256784f8507`
- Overall gate: FAIL
- Failed gates: answer_accuracy, citation_accuracy, latency_p95

## Metrics

| Metric | Result | Target |
| --- | ---: | ---: |
| Answer accuracy | 22.22% | >= 95% |
| Citation accuracy | 22.22% | >= 95% |
| End-to-end pass rate | 11.11% | diagnostic |
| End-to-end p95 latency | 58.056s | <= 10.0s |

## Per-document structure


## Latency bottlenecks

- Structured Lookup: average 4.698s, p95 7.314s
- Rewrite Answer LLM: average 2.735s, p95 4.508s
- Final Answer: average 2.542s, p95 4.076s
- Rewrite Standalone Query: average 0.987s, p95 1.521s
- 지식 검색: average 0.513s, p95 0.602s
- Build Eval Log Body: average 0.068s, p95 0.141s
- Build Session Context: average 0.101s, p95 0.129s
- Build Merge Evidence Body: average 0.053s, p95 0.095s
- Build Verify Body: average 0.055s, p95 0.091s
- Parse Merge Evidence: average 0.048s, p95 0.084s

## Failed quality cases

- `g27_ai_software_quota` (answer, citation): 2027학년도 AI소프트웨어학과 모집 정원은 몇 명인가요?
- `g27_free_major_early_quota` (answer, citation): 2027학년도 자유전공학부 수시1차 일반고 모집인원은 몇 명인가요?
- `g27_regular_documents_deadline` (answer): 2027학년도 정시 해당자의 서류제출 마감은 언제인가요?
- `g27_interview_departments` (answer, citation): 2027학년도 수시 면접 대상 학과 네 곳을 알려주세요.
- `g27_school_violence_rule` (answer, citation): 학교폭력 조치사항이 학교생활기록부에 있으면 2027학년도 전형 지원이 가능한가요?
- `g27_language_score_requirement` (answer, citation): 2027학년도 특기자 어학전형의 TOEIC 또는 TOEIC Speaking 기준은 무엇인가요?
- `g27_rural_type_one` (answer, citation): 2027학년도 농어촌 유형Ⅰ의 재학 및 거주 요건을 설명해 주세요.
- `g27_school_record_batch_range` (answer, citation): 2027학년도 고교 학교생활기록부가 출신고교에서 일괄 제공되는 졸업 연도 범위는 무엇인가요?
- `g27_best_semester_rule` (answer, citation): 2027학년도 수시 학생부 교과 성적은 어느 학기 성적을 반영하나요?
- `g27_z_score_formula` (answer, citation): 표준편차가 제공되는 교과목의 Z값 산식은 무엇인가요?
- `g27_csat_best_areas` (answer, citation): 2027학년도 정시 수능 반영에서 국어·수학·탐구 중 몇 개 영역을 반영하나요?
- `g27_interview_grade_points` (answer, citation): 2027학년도 면접고사 A, B, C, D 등급 배점은 각각 몇 점인가요?
- `g27_application_fees` (answer): 2027학년도 비면접학과와 면접학과의 원서접수 비용 합계는 각각 얼마인가요?
- `g27_tuition` (answer, citation): 2027학년도 모집요강에 적힌 전 학과 1학기와 2학기 등록금은 얼마인가요?
- `g27_first_semester_leave` (citation): 2027학년도 신입생은 입학학기에 휴학할 수 있나요? 예외도 알려주세요.
- `g27_refund_amount` (answer): 2027학년도 입학포기 시 입학 전 등록금 반환금액은 얼마인가요?
- `g26_fill_rate` (answer, citation): document.pdf에 표시된 신입생 충원률은 얼마인가요?
- `g26_scholarship_per_student` (answer, citation): document.pdf에서 학생 1인당 장학금은 얼마로 소개되나요?
- `g26_dorm_capacity` (answer, citation): document.pdf의 생활관 총 수용인원은 몇 명인가요?
- `g26_foundation_year` (answer, citation): document.pdf에 적힌 개교년도는 몇 년인가요?
- `g26_free_major_history` (citation): 자유전공학부는 연혁상 몇 년에 신설되었나요?
- `g26_employment_retention` (answer, citation): document.pdf의 2023년 유지취업률은 얼마인가요?
- `g26_career_support_duration` (citation): 대학일자리플러스센터의 졸업 후 취업지원은 몇 년까지 지속되나요?
- `g26_transfer_semesters` (answer, citation): 전과 신청이 가능한 시행 학기는 언제인가요?
- `g26_cyber_university_departments` (answer, citation): 사이버 한국외국어대학교와 경희사이버대학교의 연계 편입학 대상 학과는 무엇인가요?
- `g26_internal_scholarships` (answer, citation): document.pdf에 나온 교내장학금 중 세 가지를 알려주세요.
- `g26_three_in_first_step` (answer, citation): 학생 3-IN 인증제도의 1-In 필수 프로그램은 무엇인가요?
- `g26_cafeteria` (answer): 학생식당 위치, 중식 이용시간, 전화번호를 알려주세요.
- `g26_computer_quota` (answer, citation): 2026학년도 컴퓨터정보공학과 모집 정원은 몇 명인가요?
- `g26_first_application_period` (citation): 2026학년도 수시1차 원서접수 기간과 마감시간은 언제인가요?
- `g26_tuition` (answer, citation): document.pdf에 적힌 전 학과 1학기와 2학기 등록금은 얼마인가요?
- `g26_refund_deadline_amount` (answer, citation): 2026학년도 입학포기 신청 마감과 반환금액은 무엇인가요?

## Interpretation

Boundary accuracy is a weighted structural score: 50% source heading/table/list/formula anchor recall, 25% meaningful-page coverage, and 25% valid segment-boundary precision. Answer and citation accuracy are scored independently. Expected content stays in the manifest; the evaluator contains no domain-specific field logic.
