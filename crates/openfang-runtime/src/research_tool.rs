//! Deep research analysis tool — `research_analyze`.
//!
//! Decomposes a research question into key points, infers user intent,
//! classifies the topic, identifies stakeholders and perspectives,
//! detects missing country coverage, and generates multi-language
//! search queries ranked by the agent's source policy.
//!
//! **Hybrid approach**: Semantic analysis (topic classification, intent,
//! perspectives, stakeholders) is performed by an LLM sub-call for accuracy.
//! Deterministic operations (country detection, coverage check, query generation)
//! remain rule-based. If the LLM is unavailable, falls back to rule-based
//! heuristics for all steps.
//!
//! The tool reads `source_policy` and `sources` from the agent's
//! per-tool configuration (`manifest.tools["research_analyze"].params`).

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tracing::{debug, warn};

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------

/// Execute the `research_analyze` tool.
///
/// `tool_params` comes from `AgentManifest.tools["research_analyze"].params`.
/// It MUST contain `source_policy` and `sources` sub-keys.
///
/// `llm_driver` is used for semantic analysis (topic classification, intent
/// inference, perspective mapping, stakeholder analysis). If `None`, falls
/// back to rule-based heuristics.
pub async fn execute(
    input: &serde_json::Value,
    tool_params: Option<&HashMap<String, serde_json::Value>>,
    llm_driver: Option<Arc<dyn crate::llm_driver::LlmDriver>>,
) -> Result<String, String> {
    let question = input["question"]
        .as_str()
        .ok_or("research_analyze requires a 'question' string")?;
    let context = input["context"].as_str();
    let depth = input["depth"].as_str().unwrap_or("standard");

    let params =
        tool_params.ok_or("research_analyze requires source_policy in agent tools config")?;

    let source_policy = params
        .get("source_policy")
        .ok_or("Missing 'source_policy' in research_analyze tool params")?;
    let sources = params
        .get("sources")
        .ok_or("Missing 'sources' in research_analyze tool params")?;

    debug!(question, depth, "research_analyze: starting analysis");

    // Step 1-A: Try LLM-based semantic analysis (topic, intent, perspectives, stakeholders)
    let llm_analysis = if let Some(driver) = &llm_driver {
        match llm_semantic_analysis(driver, question, context, depth).await {
            Ok(analysis) => {
                debug!("research_analyze: LLM semantic analysis succeeded");
                Some(analysis)
            }
            Err(e) => {
                warn!("research_analyze: LLM analysis failed, falling back to rules: {e}");
                None
            }
        }
    } else {
        debug!("research_analyze: no LLM driver, using rule-based analysis");
        None
    };

    // Step 1-B: Extract or fall back for each semantic field
    let classification = llm_analysis
        .as_ref()
        .map(|a| a.topic_classification.clone())
        .unwrap_or_else(|| classify_topic(question));

    let key_points = llm_analysis
        .as_ref()
        .map(|a| a.key_points.clone())
        .unwrap_or_else(|| extract_key_points(question, context, depth));

    let intent = llm_analysis
        .as_ref()
        .map(|a| a.user_intent.clone())
        .unwrap_or_else(|| infer_user_intent(question, context, &classification));

    // Step 2: Country/entity extraction (always rule-based — deterministic lookup)
    let detected_countries = detect_countries(question);

    // Merge LLM-detected countries with rule-detected ones
    let all_countries = if let Some(ref analysis) = llm_analysis {
        let mut merged = detected_countries.clone();
        for s in &analysis.stakeholders {
            if !s.country_code.is_empty() && !merged.contains(&s.country_code) {
                merged.push(s.country_code.clone());
            }
        }
        merged
    } else {
        detected_countries
    };

    // Step 3: Country coverage check against source registry (always rule-based)
    let coverage = check_country_coverage(&all_countries, sources, source_policy);

    // Step 4: Stakeholders — prefer LLM output, fall back to rules
    let stakeholders = llm_analysis
        .as_ref()
        .map(|a| a.stakeholders.clone())
        .unwrap_or_else(|| identify_stakeholders(question, &all_countries, &classification));

    // Step 5: Perspectives — prefer LLM output, fall back to rules
    let perspectives = llm_analysis
        .as_ref()
        .map(|a| a.perspectives.clone())
        .unwrap_or_else(|| map_perspectives(&classification, &stakeholders));

    // Step 6: Multi-language search query generation (always rule-based — deterministic)
    let queries = generate_search_queries(
        question,
        &key_points,
        &stakeholders,
        sources,
        source_policy,
        depth,
    );

    // Assemble output
    let result = AnalysisResult {
        question_original: question.to_string(),
        key_points,
        user_intent: intent,
        topic_classification: classification,
        perspectives,
        stakeholders,
        country_coverage: coverage,
        search_queries: queries,
    };

    serde_json::to_string_pretty(&result).map_err(|e| format!("JSON serialization failed: {e}"))
}

// ---------------------------------------------------------------------------
// Output types
// ---------------------------------------------------------------------------

#[derive(Debug, Serialize, Deserialize)]
pub struct AnalysisResult {
    pub question_original: String,
    pub key_points: Vec<KeyPoint>,
    pub user_intent: UserIntent,
    pub topic_classification: TopicClassification,
    pub perspectives: Vec<Perspective>,
    pub stakeholders: Vec<Stakeholder>,
    pub country_coverage: CountryCoverage,
    pub search_queries: Vec<SearchQuery>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct KeyPoint {
    pub point: String,
    pub importance: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserIntent {
    pub primary_goal: String,
    pub secondary_goals: Vec<String>,
    pub assumed_background: String,
}

impl Default for UserIntent {
    fn default() -> Self {
        Self {
            primary_goal: "Understand the topic".to_string(),
            secondary_goals: vec![],
            assumed_background: "General audience".to_string(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TopicClassification {
    pub domain: String,
    pub is_social_issue: bool,
    pub is_controversial: bool,
    pub temporal_relevance: String,
}

impl Default for TopicClassification {
    fn default() -> Self {
        Self {
            domain: "general".to_string(),
            is_social_issue: false,
            is_controversial: false,
            temporal_relevance: "both".to_string(),
        }
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Perspective {
    pub viewpoint: String,
    pub typical_holders: String,
    pub key_arguments: Vec<String>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Stakeholder {
    pub name: String,
    pub country_code: String,
    #[serde(rename = "type")]
    pub stakeholder_type: String,
    pub interest: String,
    pub position: String,
    pub primary_language: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct CountryCoverage {
    pub known_countries: Vec<String>,
    pub missing_countries: Vec<String>,
    pub bootstrap_required: bool,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct SearchQuery {
    pub query: String,
    pub language: String,
    pub target_country: String,
    pub target_category: String,
    pub priority_domains: Vec<String>,
    pub pool_tier: String,
}

// ---------------------------------------------------------------------------
// LLM-based semantic analysis
// ---------------------------------------------------------------------------

/// Result from LLM semantic analysis — covers the fields that benefit from
/// language understanding rather than rule-based matching.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct LlmSemanticResult {
    #[serde(default)]
    topic_classification: TopicClassification,
    #[serde(default)]
    key_points: Vec<KeyPoint>,
    #[serde(default)]
    user_intent: UserIntent,
    #[serde(default)]
    perspectives: Vec<Perspective>,
    #[serde(default)]
    stakeholders: Vec<Stakeholder>,
}

/// System prompt for the semantic analysis sub-call.
const SEMANTIC_ANALYSIS_SYSTEM: &str = "\
You are a research question analyzer. Given a question and optional context, \
produce a JSON object with exactly these fields (no markdown, no explanation, \
pure JSON only):

{
  \"topic_classification\": {
    \"domain\": \"geopolitics|economics|ideology|social|technology|energy_climate|general\",
    \"is_social_issue\": true/false,
    \"is_controversial\": true/false,
    \"temporal_relevance\": \"current|historical|both\"
  },
  \"key_points\": [
    {\"point\": \"...\", \"importance\": \"high|medium|low\"}
  ],
  \"user_intent\": {
    \"primary_goal\": \"...\",
    \"secondary_goals\": [\"...\"],
    \"assumed_background\": \"...\"
  },
  \"perspectives\": [
    {
      \"viewpoint\": \"...\",
      \"typical_holders\": \"...\",
      \"key_arguments\": [\"...\"]
    }
  ],
  \"stakeholders\": [
    {
      \"name\": \"...\",
      \"country_code\": \"XX (ISO 2-letter)\",
      \"type\": \"government|corporation|ngo|igo|public|media\",
      \"interest\": \"...\",
      \"position\": \"See search results\",
      \"primary_language\": \"en|zh|ru|fr|de|es|ar|ja|ko|fa|he|it|nl|id|hi|ur|tr|pt\"
    }
  ]
}

Rules:
- domain must be one of: geopolitics, economics, ideology, social, technology, energy_climate, general
- For topics like \"chip war\" or \"芯片战争\", classify as economics (not geopolitics) because the core subject is trade/technology
- is_social_issue = true for geopolitics, economics, ideology, social, energy_climate
- Identify ALL relevant countries/stakeholders, even if not explicitly named
- key_points should decompose the question into 2-5 analytical sub-points
- perspectives should include at least 2 distinct viewpoints for controversial topics
- Output ONLY valid JSON, no other text";

/// Perform semantic analysis via a single LLM sub-call.
async fn llm_semantic_analysis(
    driver: &Arc<dyn crate::llm_driver::LlmDriver>,
    question: &str,
    context: Option<&str>,
    depth: &str,
) -> Result<LlmSemanticResult, String> {
    use crate::llm_driver::{CompletionRequest, CompletionResponse};
    use openfang_types::message::Message;

    let user_msg = if let Some(ctx) = context {
        format!(
            "Analyze this research question (depth: {depth}):\n\nQuestion: {question}\nContext: {ctx}"
        )
    } else {
        format!("Analyze this research question (depth: {depth}):\n\nQuestion: {question}")
    };

    let request = CompletionRequest {
        model: String::new(), // Use whatever model the driver is configured with
        messages: vec![Message::user(&user_msg)],
        tools: vec![],
        max_tokens: 2048,
        temperature: 0.2, // Low temperature for structured output
        system: Some(SEMANTIC_ANALYSIS_SYSTEM.to_string()),
        thinking: None,
    };

    let response: CompletionResponse = driver
        .complete(request)
        .await
        .map_err(|e| format!("LLM sub-call failed: {e}"))?;

    let text = response.text();
    if text.trim().is_empty() {
        return Err("LLM returned empty response".to_string());
    }

    // Try to parse the JSON — handle cases where LLM wraps it in ```json blocks
    let json_str = extract_json_from_response(&text);

    serde_json::from_str::<LlmSemanticResult>(json_str)
        .map_err(|e| format!("Failed to parse LLM JSON output: {e}\nRaw: {}", &text[..text.len().min(500)]))
}

/// Extract JSON from LLM response, handling markdown code fences.
fn extract_json_from_response(text: &str) -> &str {
    let trimmed = text.trim();

    // Try to extract from ```json ... ``` blocks
    if let Some(start) = trimmed.find("```json") {
        let after = &trimmed[start + 7..];
        if let Some(end) = after.find("```") {
            return after[..end].trim();
        }
    }
    // Try ``` ... ``` without language tag
    if let Some(start) = trimmed.find("```") {
        let after = &trimmed[start + 3..];
        if let Some(end) = after.find("```") {
            return after[..end].trim();
        }
    }
    // Try to find a JSON object directly
    if let Some(start) = trimmed.find('{') {
        if let Some(end) = trimmed.rfind('}') {
            if end > start {
                return &trimmed[start..=end];
            }
        }
    }
    trimmed
}

// ---------------------------------------------------------------------------
// Rule-based fallbacks (Step 1: Topic classification)
// ---------------------------------------------------------------------------

fn classify_topic(question: &str) -> TopicClassification {
    let q = question.to_lowercase();

    // Check economics FIRST — its keywords (chip, semiconductor, trade, tariff)
    // are more specific than geopolitics keywords (war, conflict) which may
    // co-occur in economic topics like "chip war" / "芯片战争".
    let domain = if matches_any(
        &q,
        &[
            "trade ",
            "tariff",
            "gdp",
            "inflation",
            "currency",
            "export",
            "import",
            "supply chain",
            "economic",
            "recession",
            "debt",
            "investment",
            "semiconductor",
            "chip",
            "decoupling",
            // Chinese terms
            "贸易",
            "关税",
            "通胀",
            "货币",
            "出口",
            "进口",
            "供应链",
            "经济",
            "衰退",
            "债务",
            "投资",
            "芯片",
            "半导体",
            "脱钩",
        ],
    ) {
        "economics"
    } else if matches_any(
        &q,
        &[
            "geopoliti",
            "sanction",
            "diplomacy",
            "territory",
            "sovereignty",
            "nato",
            "military",
            "war ",
            "civil war",
            "conflict",
            "alliance",
            "treaty",
            "invasion",
            "annex",
            "border dispute",
            "arms race",
            // Chinese terms
            "地缘",
            "制裁",
            "外交",
            "领土",
            "主权",
            "军事",
            "战争",
            "冲突",
            "同盟",
            "条约",
            "入侵",
            "边界",
            "军备",
        ],
    ) {
        "geopolitics"
    } else if matches_any(
        &q,
        &[
            "ideolog",
            "democra",
            "authorit",
            "communi",
            "capitali",
            "liberal",
            "conservat",
            "propagan",
            "censor",
            "freedom",
            "human rights",
            // Chinese terms
            "意识形态",
            "民主",
            "威权",
            "共产",
            "资本",
            "自由",
            "保守",
            "宣传",
            "审查",
            "人权",
        ],
    ) {
        "ideology"
    } else if matches_any(
        &q,
        &[
            "public opinion",
            "social media",
            "protest",
            "inequality",
            "immigration",
            "refugee",
            "populis",
            "polariz",
            "disinformation",
            "narrative",
            // Chinese terms
            "舆论",
            "社交媒体",
            "抗议",
            "不平等",
            "移民",
            "难民",
            "民粹",
            "极化",
            "虚假信息",
            "叙事",
            "矛盾",
        ],
    ) {
        "social"
    } else if matches_any(
        &q,
        &[
            "ai ",
            "artificial intelligence",
            "cyber",
            "quantum",
            "5g",
            "tech",
            "digital",
            "biotech",
            "space",
            "satellite",
            // Chinese terms
            "人工智能",
            "网络",
            "量子",
            "技术",
            "数字",
            "生物",
            "太空",
            "卫星",
        ],
    ) {
        "technology"
    } else if matches_any(
        &q,
        &[
            "climate",
            "energy",
            "oil",
            "gas",
            "carbon",
            "emission",
            "renewable",
            "nuclear",
            "opec",
            // Chinese terms
            "气候",
            "能源",
            "石油",
            "天然气",
            "碳",
            "排放",
            "可再生",
            "核能",
        ],
    ) {
        "energy_climate"
    } else {
        "general"
    };

    let is_social_issue = matches!(
        domain,
        "geopolitics" | "economics" | "ideology" | "social" | "energy_climate"
    );

    let is_controversial = is_social_issue
        && matches_any(
            &q,
            &[
                "conflict",
                "dispute",
                "war",
                "sanction",
                "controversy",
                "debate",
                "crisis",
                "oppose",
                "tension",
                "冲突",
                "争端",
                "战争",
                "制裁",
                "争议",
                "危机",
                "对立",
                "紧张",
            ],
        );

    let temporal_relevance = if matches_any(
        &q,
        &[
            "2025", "2026", "latest", "recent", "current", "now", "today", "最新", "近期", "当前",
            "目前",
        ],
    ) {
        "current"
    } else if matches_any(
        &q,
        &[
            "history",
            "historical",
            "origin",
            "since",
            "evolution",
            "历史",
            "起源",
            "演变",
        ],
    ) {
        "historical"
    } else {
        "both"
    };

    TopicClassification {
        domain: domain.to_string(),
        is_social_issue,
        is_controversial,
        temporal_relevance: temporal_relevance.to_string(),
    }
}

// ---------------------------------------------------------------------------
// Step 2: Country detection
// ---------------------------------------------------------------------------

/// Mapping of country names/aliases to ISO 2-letter codes.
/// Covers the 24 seed countries plus common aliases in English and Chinese.
fn country_map() -> Vec<(&'static str, &'static str)> {
    vec![
        // CN
        ("china", "CN"),
        ("chinese", "CN"),
        ("beijing", "CN"),
        ("中国", "CN"),
        ("北京", "CN"),
        ("中美", "CN"),
        ("中日", "CN"),
        ("中俄", "CN"),
        ("中欧", "CN"),
        // US
        ("united states", "US"),
        ("america", "US"),
        ("washington", "US"),
        ("u.s.", "US"),
        ("us-china", "US"),
        ("us-", "US"),
        ("美国", "US"),
        ("华盛顿", "US"),
        ("中美", "US"),
        ("美中", "US"),
        // UK
        ("united kingdom", "UK"),
        ("britain", "UK"),
        ("british", "UK"),
        ("london", "UK"),
        ("英国", "UK"),
        ("伦敦", "UK"),
        // CA
        ("canada", "CA"),
        ("canadian", "CA"),
        ("ottawa", "CA"),
        ("加拿大", "CA"),
        // FR
        ("france", "FR"),
        ("french", "FR"),
        ("paris", "FR"),
        ("法国", "FR"),
        ("巴黎", "FR"),
        // DE
        ("germany", "DE"),
        ("german", "DE"),
        ("berlin", "DE"),
        ("德国", "DE"),
        ("柏林", "DE"),
        // IT
        ("italy", "IT"),
        ("italian", "IT"),
        ("rome", "IT"),
        ("意大利", "IT"),
        ("罗马", "IT"),
        // ES
        ("spain", "ES"),
        ("spanish", "ES"),
        ("madrid", "ES"),
        ("西班牙", "ES"),
        // NL
        ("netherlands", "NL"),
        ("dutch", "NL"),
        ("holland", "NL"),
        ("荷兰", "NL"),
        // RU
        ("russia", "RU"),
        ("russian", "RU"),
        ("moscow", "RU"),
        ("kremlin", "RU"),
        ("俄罗斯", "RU"),
        ("莫斯科", "RU"),
        ("克里姆林宫", "RU"),
        // SA
        ("saudi", "SA"),
        ("saudi arabia", "SA"),
        ("riyadh", "SA"),
        ("沙特", "SA"),
        // AE
        ("uae", "AE"),
        ("emirates", "AE"),
        ("dubai", "AE"),
        ("abu dhabi", "AE"),
        ("阿联酋", "AE"),
        ("迪拜", "AE"),
        // QA
        ("qatar", "QA"),
        ("doha", "QA"),
        ("卡塔尔", "QA"),
        ("多哈", "QA"),
        // IR
        ("iran", "IR"),
        ("iranian", "IR"),
        ("tehran", "IR"),
        ("伊朗", "IR"),
        ("德黑兰", "IR"),
        // IL
        ("israel", "IL"),
        ("israeli", "IL"),
        ("tel aviv", "IL"),
        ("jerusalem", "IL"),
        ("以色列", "IL"),
        ("耶路撒冷", "IL"),
        // JP
        ("japan", "JP"),
        ("japanese", "JP"),
        ("tokyo", "JP"),
        ("日本", "JP"),
        ("东京", "JP"),
        // KR
        ("south korea", "KR"),
        ("korean", "KR"),
        ("seoul", "KR"),
        ("韩国", "KR"),
        ("首尔", "KR"),
        // KP
        ("north korea", "KP"),
        ("pyongyang", "KP"),
        ("dprk", "KP"),
        ("朝鲜", "KP"),
        ("平壤", "KP"),
        // SG
        ("singapore", "SG"),
        ("新加坡", "SG"),
        // ID
        ("indonesia", "ID"),
        ("indonesian", "ID"),
        ("jakarta", "ID"),
        ("印尼", "ID"),
        ("印度尼西亚", "ID"),
        ("雅加达", "ID"),
        // IN
        ("india", "IN"),
        ("indian", "IN"),
        ("new delhi", "IN"),
        ("delhi", "IN"),
        ("印度", "IN"),
        ("新德里", "IN"),
        // PK
        ("pakistan", "PK"),
        ("pakistani", "PK"),
        ("islamabad", "PK"),
        ("巴基斯坦", "PK"),
        // TR
        ("turkey", "TR"),
        ("turkish", "TR"),
        ("türkiye", "TR"),
        ("ankara", "TR"),
        ("土耳其", "TR"),
        ("安卡拉", "TR"),
        // MX
        ("mexico", "MX"),
        ("mexican", "MX"),
        ("墨西哥", "MX"),
        // EU / International orgs (mapped as special codes)
        ("european union", "EU"),
        ("eu ", "EU"),
        ("欧盟", "EU"),
        ("nato", "NATO"),
        ("un ", "UN"),
        ("united nations", "UN"),
        ("联合国", "UN"),
    ]
}

fn detect_countries(question: &str) -> Vec<String> {
    let q_lower = question.to_lowercase();
    let mut seen = std::collections::HashSet::new();
    let mut codes = Vec::new();

    for (name, code) in country_map() {
        if q_lower.contains(name) && seen.insert(code.to_string()) {
            codes.push(code.to_string());
        }
    }

    // If no countries detected, infer from topic keywords
    if codes.is_empty() {
        // Default to major global actors for general geopolitical topics
        let q = question.to_lowercase();
        if matches_any(
            &q,
            &["global", "world", "international", "全球", "世界", "国际"],
        ) {
            codes.extend(["US", "CN", "EU"].iter().map(|s| s.to_string()));
        }
    }

    codes
}

// ---------------------------------------------------------------------------
// Step 3: Country coverage check
// ---------------------------------------------------------------------------

fn check_country_coverage(
    detected: &[String],
    sources: &serde_json::Value,
    source_policy: &serde_json::Value,
) -> CountryCoverage {
    let sources_map = sources.as_object();
    let min_sources = source_policy
        .get("country_min_effective_sources")
        .and_then(|v| v.as_u64())
        .unwrap_or(10) as usize;

    let mut known = Vec::new();
    let mut missing = Vec::new();

    for code in detected {
        // Skip non-country codes (EU, NATO, UN)
        if matches!(code.as_str(), "EU" | "NATO" | "UN") {
            known.push(code.clone());
            continue;
        }

        let has_sources = sources_map
            .and_then(|m| m.get(code))
            .map(|country_sources| count_effective_sources(country_sources) >= min_sources)
            .unwrap_or(false);

        if has_sources {
            known.push(code.clone());
        } else {
            missing.push(code.clone());
        }
    }

    let bootstrap_required = !missing.is_empty();

    CountryCoverage {
        known_countries: known,
        missing_countries: missing,
        bootstrap_required,
    }
}

/// Count effective (non-empty) sources across all categories for a country.
fn count_effective_sources(country_sources: &serde_json::Value) -> usize {
    let mut count = 0;
    if let Some(obj) = country_sources.as_object() {
        for (key, val) in obj {
            if key == "lang" {
                continue;
            }
            if let Some(arr) = val.as_array() {
                count += arr
                    .iter()
                    .filter(|v| !v.as_str().unwrap_or("").is_empty())
                    .count();
            }
        }
    }
    count
}

// ---------------------------------------------------------------------------
// Step 4: Key points extraction
// ---------------------------------------------------------------------------

fn extract_key_points(question: &str, context: Option<&str>, depth: &str) -> Vec<KeyPoint> {
    let mut points = Vec::new();

    // Split on common delimiters to find sub-questions
    let parts: Vec<&str> = question
        .split(['?', '？', ',', '，', ';', '；'])
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .collect();

    if parts.len() > 1 {
        // Multiple sub-questions — each is a key point
        for (i, part) in parts.iter().enumerate() {
            points.push(KeyPoint {
                point: part.to_string(),
                importance: if i == 0 { "high" } else { "medium" }.to_string(),
            });
        }
    } else {
        // Single question — decompose by interrogative words
        points.push(KeyPoint {
            point: question.to_string(),
            importance: "high".to_string(),
        });
    }

    // Add context-derived point if available
    if let Some(ctx) = context {
        if !ctx.trim().is_empty() {
            points.push(KeyPoint {
                point: format!("Context: {ctx}"),
                importance: "medium".to_string(),
            });
        }
    }

    // For deep analysis, add analytical sub-points
    if depth == "deep" {
        points.push(KeyPoint {
            point: "Historical background and root causes".to_string(),
            importance: "medium".to_string(),
        });
        points.push(KeyPoint {
            point: "Future trajectory and potential scenarios".to_string(),
            importance: "medium".to_string(),
        });
    }

    points
}

// ---------------------------------------------------------------------------
// Step 5: Stakeholder identification
// ---------------------------------------------------------------------------

fn identify_stakeholders(
    question: &str,
    detected_countries: &[String],
    classification: &TopicClassification,
) -> Vec<Stakeholder> {
    let mut stakeholders = Vec::new();

    // Map country codes to stakeholder entries
    for code in detected_countries {
        if let Some((name, lang, stype)) = country_stakeholder_info(code) {
            stakeholders.push(Stakeholder {
                name: name.to_string(),
                country_code: code.clone(),
                stakeholder_type: stype.to_string(),
                interest: infer_interest(code, question),
                position: "See search results".to_string(),
                primary_language: lang.to_string(),
            });
        }
    }

    // For social/geopolitical issues, add international org stakeholders
    if classification.is_social_issue && stakeholders.len() >= 2 {
        let q = question.to_lowercase();
        if matches_any(&q, &["trade", "tariff", "贸易", "关税"]) {
            stakeholders.push(Stakeholder {
                name: "WTO / World Trade Organization".to_string(),
                country_code: "INT".to_string(),
                stakeholder_type: "igo".to_string(),
                interest: "Rules-based trade order".to_string(),
                position: "See search results".to_string(),
                primary_language: "en".to_string(),
            });
        }
        if matches_any(
            &q,
            &["human rights", "人权", "refugee", "难民", "humanitarian"],
        ) {
            stakeholders.push(Stakeholder {
                name: "UN / United Nations".to_string(),
                country_code: "INT".to_string(),
                stakeholder_type: "igo".to_string(),
                interest: "Humanitarian norms and international law".to_string(),
                position: "See search results".to_string(),
                primary_language: "en".to_string(),
            });
        }
    }

    stakeholders
}

/// Returns (display_name, primary_language, stakeholder_type) for a country code.
fn country_stakeholder_info(code: &str) -> Option<(&'static str, &'static str, &'static str)> {
    match code {
        "CN" => Some(("China", "zh", "government")),
        "US" => Some(("United States", "en", "government")),
        "UK" => Some(("United Kingdom", "en", "government")),
        "CA" => Some(("Canada", "en", "government")),
        "FR" => Some(("France", "fr", "government")),
        "DE" => Some(("Germany", "de", "government")),
        "IT" => Some(("Italy", "it", "government")),
        "ES" => Some(("Spain", "es", "government")),
        "NL" => Some(("Netherlands", "nl", "government")),
        "RU" => Some(("Russia", "ru", "government")),
        "SA" => Some(("Saudi Arabia", "ar", "government")),
        "AE" => Some(("UAE", "ar", "government")),
        "QA" => Some(("Qatar", "ar", "government")),
        "IR" => Some(("Iran", "fa", "government")),
        "IL" => Some(("Israel", "he", "government")),
        "JP" => Some(("Japan", "ja", "government")),
        "KR" => Some(("South Korea", "ko", "government")),
        "KP" => Some(("North Korea", "ko", "government")),
        "SG" => Some(("Singapore", "en", "government")),
        "ID" => Some(("Indonesia", "id", "government")),
        "IN" => Some(("India", "hi", "government")),
        "PK" => Some(("Pakistan", "ur", "government")),
        "TR" => Some(("Turkey", "tr", "government")),
        "MX" => Some(("Mexico", "es", "government")),
        "EU" => Some(("European Union", "en", "igo")),
        "NATO" => Some(("NATO", "en", "igo")),
        "UN" => Some(("United Nations", "en", "igo")),
        _ => None,
    }
}

fn infer_interest(country_code: &str, question: &str) -> String {
    let q = question.to_lowercase();
    // Simple heuristic: generate a one-liner interest description
    if matches_any(&q, &["trade", "tariff", "economic", "贸易", "关税", "经济"]) {
        format!("Economic interests and trade policy of {country_code}")
    } else if matches_any(
        &q,
        &["military", "security", "defense", "军事", "安全", "防务"],
    ) {
        format!("National security interests of {country_code}")
    } else if matches_any(
        &q,
        &[
            "technology",
            "chip",
            "semiconductor",
            "ai",
            "技术",
            "芯片",
            "半导体",
        ],
    ) {
        format!("Technology competitiveness of {country_code}")
    } else {
        format!("Strategic interests of {country_code}")
    }
}

// ---------------------------------------------------------------------------
// Step 6: Perspective mapping
// ---------------------------------------------------------------------------

fn map_perspectives(
    classification: &TopicClassification,
    stakeholders: &[Stakeholder],
) -> Vec<Perspective> {
    let mut perspectives = Vec::new();

    if stakeholders.len() >= 2 {
        // Generate perspectives based on stakeholder pairs
        for stakeholder in stakeholders.iter().take(5) {
            perspectives.push(Perspective {
                viewpoint: format!("{} perspective", stakeholder.name),
                typical_holders: format!("Government, media, and public in {}", stakeholder.name),
                key_arguments: vec![format!(
                    "Arguments aligned with {} national interest",
                    stakeholder.name
                )],
            });
        }
    }

    // Always add a neutral/analytical perspective
    if classification.is_social_issue {
        perspectives.push(Perspective {
            viewpoint: "International / neutral analytical perspective".to_string(),
            typical_holders: "International organizations, neutral think tanks, academics"
                .to_string(),
            key_arguments: vec![
                "Evidence-based analysis beyond national narratives".to_string(),
                "Focus on systemic factors and long-term trends".to_string(),
            ],
        });
    }

    perspectives
}

// ---------------------------------------------------------------------------
// Step 7: Search query generation
// ---------------------------------------------------------------------------

fn generate_search_queries(
    question: &str,
    key_points: &[KeyPoint],
    stakeholders: &[Stakeholder],
    sources: &serde_json::Value,
    source_policy: &serde_json::Value,
    depth: &str,
) -> Vec<SearchQuery> {
    let mut queries = Vec::new();
    let sources_map = sources.as_object();

    let primary_pool_size = source_policy
        .get("search")
        .and_then(|s| s.get("primary_pool_size"))
        .and_then(|v| v.as_u64())
        .unwrap_or(10) as usize;

    // Determine category search order based on tiers
    let search_categories = [
        ("official_media", "tier1"),
        ("think_tank", "tier2"),
        ("mainstream_media", "tier3"),
        ("regional_media", "tier4"),
        ("academic", "tier2"),
    ];

    for stakeholder in stakeholders {
        let code = &stakeholder.country_code;
        let lang = &stakeholder.primary_language;

        // Get the source domains for this country
        let country_domains: Vec<String> = sources_map
            .and_then(|m| m.get(code))
            .map(|cs| {
                let mut domains = Vec::new();
                for (cat, _) in &search_categories {
                    if let Some(arr) = cs.get(*cat).and_then(|v| v.as_array()) {
                        for d in arr {
                            if let Some(s) = d.as_str() {
                                if !s.is_empty() {
                                    domains.push(s.to_string());
                                }
                            }
                        }
                    }
                }
                domains
            })
            .unwrap_or_default();

        // Take the top N as priority domains
        let priority: Vec<String> = country_domains
            .iter()
            .take(primary_pool_size)
            .cloned()
            .collect();

        // Primary query: the question in stakeholder's language context
        queries.push(SearchQuery {
            query: format_query_for_language(question, lang),
            language: lang.clone(),
            target_country: code.clone(),
            target_category: "mainstream_media".to_string(),
            priority_domains: priority.clone(),
            pool_tier: "top10".to_string(),
        });

        // For deep analysis, add category-specific queries
        if depth == "deep" || depth == "standard" {
            for (cat, _) in &search_categories {
                let cat_domains: Vec<String> = sources_map
                    .and_then(|m| m.get(code))
                    .and_then(|cs| cs.get(*cat))
                    .and_then(|v| v.as_array())
                    .map(|arr| {
                        arr.iter()
                            .filter_map(|v| v.as_str())
                            .filter(|s| !s.is_empty())
                            .map(String::from)
                            .collect()
                    })
                    .unwrap_or_default();

                if !cat_domains.is_empty() {
                    // Extract the core topic for more targeted search
                    let core_topic = extract_core_topic(question);
                    queries.push(SearchQuery {
                        query: format!(
                            "{} site:{}",
                            format_query_for_language(&core_topic, lang),
                            cat_domains.first().unwrap_or(&String::new())
                        ),
                        language: lang.clone(),
                        target_country: code.clone(),
                        target_category: cat.to_string(),
                        priority_domains: cat_domains,
                        pool_tier: "top10".to_string(),
                    });
                }
            }
        }

        // Add high-importance key point queries
        for kp in key_points.iter().filter(|k| k.importance == "high") {
            if kp.point != question && !kp.point.starts_with("Context:") {
                queries.push(SearchQuery {
                    query: format_query_for_language(&kp.point, lang),
                    language: lang.clone(),
                    target_country: code.clone(),
                    target_category: "think_tank".to_string(),
                    priority_domains: priority.clone(),
                    pool_tier: "top10".to_string(),
                });
            }
        }
    }

    queries
}

/// Format a query string for a target language.
/// For non-English/non-Chinese queries, append the language name as a hint
/// (actual translation is left to the LLM agent in the search phase).
fn format_query_for_language(query: &str, lang: &str) -> String {
    match lang {
        "en" | "zh" => query.to_string(),
        "fr" => format!("{query} (en français)"),
        "de" => format!("{query} (auf Deutsch)"),
        "es" => format!("{query} (en español)"),
        "ru" => format!("{query} (на русском)"),
        "ar" => format!("{query} (بالعربية)"),
        "ja" => format!("{query} (日本語で)"),
        "ko" => format!("{query} (한국어로)"),
        "fa" => format!("{query} (به فارسی)"),
        "he" => format!("{query} (בעברית)"),
        "it" => format!("{query} (in italiano)"),
        "nl" => format!("{query} (in het Nederlands)"),
        "id" => format!("{query} (dalam bahasa Indonesia)"),
        "hi" => format!("{query} (हिंदी में)"),
        "ur" => format!("{query} (اردو میں)"),
        "tr" => format!("{query} (Türkçe)"),
        "pt" => format!("{query} (em português)"),
        _ => query.to_string(),
    }
}

/// Extract the core topic from a question, stripping interrogatives.
fn extract_core_topic(question: &str) -> String {
    let q = question.trim();
    // Strip common question prefixes
    let stripped = q
        .trim_start_matches("What is ")
        .trim_start_matches("What are ")
        .trim_start_matches("How does ")
        .trim_start_matches("How do ")
        .trim_start_matches("Why is ")
        .trim_start_matches("Why are ")
        .trim_start_matches("What impact ")
        .trim_start_matches("What effect ")
        .trim_end_matches('?')
        .trim_end_matches('？');
    if stripped.len() < q.len() {
        stripped.to_string()
    } else {
        q.to_string()
    }
}

// ---------------------------------------------------------------------------
// Step 8: User intent inference
// ---------------------------------------------------------------------------

fn infer_user_intent(
    question: &str,
    context: Option<&str>,
    classification: &TopicClassification,
) -> UserIntent {
    let q = question.to_lowercase();

    let primary_goal = if matches_any(
        &q,
        &[
            "impact",
            "effect",
            "influence",
            "consequence",
            "影响",
            "后果",
            "效果",
        ],
    ) {
        "Understand the impact and consequences of a development".to_string()
    } else if matches_any(&q, &["why", "reason", "cause", "为什么", "原因", "缘由"]) {
        "Understand the root causes and motivations".to_string()
    } else if matches_any(&q, &["how", "mechanism", "process", "如何", "怎样", "机制"]) {
        "Understand the mechanism or process".to_string()
    } else if matches_any(
        &q,
        &[
            "future", "predict", "forecast", "trend", "未来", "预测", "趋势",
        ],
    ) {
        "Forecast future developments and scenarios".to_string()
    } else if matches_any(
        &q,
        &[
            "compare",
            "difference",
            "versus",
            "vs",
            "比较",
            "区别",
            "对比",
        ],
    ) {
        "Compare different positions or approaches".to_string()
    } else {
        "Obtain a comprehensive understanding of the topic".to_string()
    };

    let mut secondary_goals = Vec::new();
    if classification.is_controversial {
        secondary_goals.push("Understand different stakeholder positions".to_string());
    }
    if classification.is_social_issue {
        secondary_goals.push("Assess geopolitical and social implications".to_string());
    }
    secondary_goals.push("Get authoritative and up-to-date information".to_string());

    let assumed_background = if let Some(ctx) = context {
        format!("User provided context: {ctx}")
    } else if classification.is_social_issue {
        "User likely follows international affairs and seeks multi-perspective analysis".to_string()
    } else {
        "General audience seeking informed overview".to_string()
    };

    UserIntent {
        primary_goal,
        secondary_goals,
        assumed_background,
    }
}

// ---------------------------------------------------------------------------
// Utility
// ---------------------------------------------------------------------------

fn matches_any(text: &str, patterns: &[&str]) -> bool {
    patterns.iter().any(|p| text.contains(p))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_classify_geopolitics() {
        let c = classify_topic("What are the geopolitical implications of NATO expansion?");
        assert_eq!(c.domain, "geopolitics");
        assert!(c.is_social_issue);
    }

    #[test]
    fn test_classify_economics() {
        let c = classify_topic("中美芯片战争对全球半导体供应链的影响");
        assert_eq!(c.domain, "economics");
        assert!(c.is_social_issue);
    }

    #[test]
    fn test_classify_ideology() {
        let c = classify_topic("How does censorship affect freedom of expression?");
        assert_eq!(c.domain, "ideology");
        assert!(c.is_social_issue);
    }

    #[test]
    fn test_classify_general() {
        let c = classify_topic("What is the best programming language for web development?");
        assert_eq!(c.domain, "general");
        assert!(!c.is_social_issue);
    }

    #[test]
    fn test_classify_temporal_current() {
        let c = classify_topic("What are the latest developments in 2026 trade negotiations?");
        assert_eq!(c.temporal_relevance, "current");
    }

    #[test]
    fn test_detect_countries_english() {
        let countries = detect_countries("US-China trade war impact on Japan and South Korea");
        assert!(countries.contains(&"US".to_string()));
        assert!(countries.contains(&"CN".to_string()));
        assert!(countries.contains(&"JP".to_string()));
        assert!(countries.contains(&"KR".to_string()));
    }

    #[test]
    fn test_detect_countries_chinese() {
        let countries = detect_countries("中美芯片战争对日本和韩国的影响");
        assert!(countries.contains(&"CN".to_string()));
        assert!(countries.contains(&"US".to_string()));
        assert!(countries.contains(&"JP".to_string()));
        assert!(countries.contains(&"KR".to_string()));
    }

    #[test]
    fn test_detect_countries_none() {
        let countries = detect_countries("global economic trends");
        // Should infer major actors for global topics
        assert!(!countries.is_empty());
    }

    #[test]
    fn test_country_coverage_known() {
        let sources = serde_json::json!({
            "US": {
                "lang": "en",
                "official_media": ["state.gov", "defense.gov", "whitehouse.gov"],
                "think_tank": ["brookings.edu", "cfr.org", "csis.org"],
                "mainstream_media": ["reuters.com", "apnews.com"],
                "regional_media": ["latimes.com"],
                "academic": ["nber.org"]
            }
        });
        let policy = serde_json::json!({ "country_min_effective_sources": 10 });
        let coverage = check_country_coverage(&["US".to_string()], &sources, &policy);
        assert!(coverage.known_countries.contains(&"US".to_string()));
        assert!(coverage.missing_countries.is_empty());
        assert!(!coverage.bootstrap_required);
    }

    #[test]
    fn test_country_coverage_missing() {
        let sources = serde_json::json!({});
        let policy = serde_json::json!({ "country_min_effective_sources": 10 });
        let coverage = check_country_coverage(&["BR".to_string()], &sources, &policy);
        assert!(coverage.missing_countries.contains(&"BR".to_string()));
        assert!(coverage.bootstrap_required);
    }

    #[test]
    fn test_count_effective_sources() {
        let cs = serde_json::json!({
            "lang": "en",
            "official_media": ["a.gov", "b.gov"],
            "think_tank": ["c.org"],
            "mainstream_media": ["d.com", "e.com", "f.com"],
            "regional_media": [],
            "influencer": ["x.com/someone"],
            "academic": ["uni.edu"]
        });
        assert_eq!(count_effective_sources(&cs), 8);
    }

    #[test]
    fn test_extract_key_points_multi() {
        let kps = extract_key_points("What is the cause? What is the effect?", None, "standard");
        assert!(kps.len() >= 2);
        assert_eq!(kps[0].importance, "high");
    }

    #[test]
    fn test_format_query_language() {
        let q = format_query_for_language("trade war", "fr");
        assert!(q.contains("français"));

        let q_en = format_query_for_language("trade war", "en");
        assert_eq!(q_en, "trade war");
    }

    #[test]
    fn test_extract_core_topic() {
        assert_eq!(
            extract_core_topic("What is the impact of tariffs?"),
            "the impact of tariffs"
        );
        assert_eq!(extract_core_topic("中美芯片战争"), "中美芯片战争");
    }

    #[tokio::test]
    async fn test_full_execute() {
        let input = serde_json::json!({
            "question": "中美芯片战争对全球半导体产业链的影响"
        });
        let mut params = HashMap::new();
        params.insert(
            "source_policy".to_string(),
            serde_json::json!({
                "country_min_effective_sources": 10,
                "country_max_effective_sources": 50,
                "search": {
                    "primary_pool_size": 10
                }
            }),
        );
        params.insert(
            "sources".to_string(),
            serde_json::json!({
                "CN": {
                    "lang": "zh",
                    "official_media": ["xinhuanet.com", "people.com.cn"],
                    "think_tank": ["cicir.ac.cn", "cass.cn"],
                    "mainstream_media": ["caixin.com", "globaltimes.cn"],
                    "regional_media": ["thepaper.cn", "bjnews.com.cn"],
                    "influencer": ["weibo.com/huxijin_gt"],
                    "academic": ["cnki.net", "nsfc.gov.cn"]
                },
                "US": {
                    "lang": "en",
                    "official_media": ["state.gov", "defense.gov", "whitehouse.gov"],
                    "think_tank": ["brookings.edu", "cfr.org", "csis.org", "rand.org"],
                    "mainstream_media": ["reuters.com", "apnews.com", "nytimes.com"],
                    "regional_media": ["latimes.com", "chicagotribune.com"],
                    "influencer": ["x.com/ianbremmer"],
                    "academic": ["nber.org", "jstor.org"]
                }
            }),
        );

        // Test with no LLM driver (rule-based fallback)
        let result = execute(&input, Some(&params), None).await;
        assert!(result.is_ok());
        let json_str = result.unwrap();
        let parsed: AnalysisResult = serde_json::from_str(&json_str).unwrap();
        assert_eq!(parsed.topic_classification.domain, "economics");
        assert!(!parsed.stakeholders.is_empty());
        assert!(!parsed.search_queries.is_empty());
        // Should have CN and US stakeholders
        assert!(parsed.stakeholders.iter().any(|s| s.country_code == "CN"));
        assert!(parsed.stakeholders.iter().any(|s| s.country_code == "US"));
    }

    #[tokio::test]
    async fn test_execute_missing_policy() {
        let input = serde_json::json!({ "question": "test" });
        let result = execute(&input, None, None).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("source_policy"));
    }

    #[tokio::test]
    async fn test_execute_missing_question() {
        let input = serde_json::json!({});
        let mut params = HashMap::new();
        params.insert("source_policy".to_string(), serde_json::json!({}));
        params.insert("sources".to_string(), serde_json::json!({}));
        let result = execute(&input, Some(&params), None).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("question"));
    }

    #[test]
    fn test_extract_json_from_response() {
        // Plain JSON
        let json = r#"{"domain": "economics"}"#;
        assert_eq!(extract_json_from_response(json), json);

        // With markdown fences
        let fenced = "```json\n{\"domain\": \"economics\"}\n```";
        assert_eq!(
            extract_json_from_response(fenced),
            "{\"domain\": \"economics\"}"
        );

        // With text before JSON
        let prefixed = "Here is the analysis:\n{\"domain\": \"economics\"}";
        assert_eq!(
            extract_json_from_response(prefixed),
            "{\"domain\": \"economics\"}"
        );
    }
}
