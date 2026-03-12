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

    // Step 6: Mode selection — prefer LLM, fall back to rules
    let analysis_mode = llm_analysis
        .as_ref()
        .filter(|a| !a.analysis_mode.is_empty())
        .map(|a| a.analysis_mode.clone())
        .unwrap_or_else(|| select_analysis_mode(&classification, question));

    // Step 7: Question type — prefer LLM, fall back to rules
    let question_type = llm_analysis
        .as_ref()
        .filter(|a| !a.question_type.is_empty())
        .map(|a| a.question_type.clone())
        .unwrap_or_else(|| infer_question_type(question, &intent));

    // Step 8: Comparison flag — prefer LLM, fall back to rules
    let comparison_required = llm_analysis
        .as_ref()
        .map(|a| a.comparison_required)
        .unwrap_or_else(|| detect_comparison_required(question));

    // Step 9: Intent flags — prefer LLM, fall back to safe defaults (all false)
    let intent_flags = llm_analysis
        .as_ref()
        .map(|a| a.intent_flags.clone())
        .unwrap_or_else(|| detect_intent_flags(question, &intent, &classification));

    // Step 10: Derive user language
    let user_lang = derive_user_language(question);

    // Step 11: Candidate frames — prefer LLM (if >= 2), fall back to rules
    let candidate_frames = {
        let llm_frames = llm_analysis
            .as_ref()
            .map(|a| a.candidate_frames.clone())
            .unwrap_or_default();

        let frames = if llm_frames.len() >= 2 {
            llm_frames
        } else if analysis_mode == "stakeholder_mode" {
            build_candidate_frames_for_stakeholder(question, &stakeholders, &classification)
        } else {
            build_candidate_frames_for_multi_hypothesis(question, &classification, &user_lang)
        };

        ensure_min_frames(frames, 2)
    };

    // Step 12: Multi-language search query generation (mode-aware)
    let (queries, search_strategy) = generate_search_queries(
        question,
        &key_points,
        &stakeholders,
        &candidate_frames,
        sources,
        source_policy,
        depth,
        &analysis_mode,
        &user_lang,
    );

    // Step 13: Generate placeholder future directions
    let future_directions = vec![
        FutureDirection {
            scenario: "Continuation of current trajectory".to_string(),
            drivers: vec!["Existing momentum and policies".to_string()],
            triggers: vec!["No major disruptive events".to_string()],
            risks: vec!["Gradual erosion if assumptions change".to_string()],
        },
        FutureDirection {
            scenario: "Significant shift or disruption".to_string(),
            drivers: vec!["Emerging pressures or breakthroughs".to_string()],
            triggers: vec!["Key event or policy change".to_string()],
            risks: vec!["Unpredictable second-order effects".to_string()],
        },
    ];

    // Assemble output
    let result = AnalysisResult {
        question_original: question.to_string(),
        key_points,
        user_intent: intent,
        topic_classification: classification,
        analysis_mode,
        question_type,
        comparison_required,
        intent_flags,
        candidate_frames,
        perspectives,
        stakeholders,
        country_coverage: coverage,
        search_strategy,
        search_queries: queries,
        future_directions,
        coverage_gaps: vec![],
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
    /// `"stakeholder_mode"` or `"multi_hypothesis_mode"`
    #[serde(default = "default_analysis_mode")]
    pub analysis_mode: String,
    /// `"open_research"` | `"feasibility"` | `"event_probability"` | `"learning_oriented"`
    #[serde(default = "default_question_type")]
    pub question_type: String,
    /// Whether the question involves explicit comparison of concepts/systems/works.
    #[serde(default)]
    pub comparison_required: bool,
    /// Conditional output flags — controls which optional report sections to produce.
    #[serde(default)]
    pub intent_flags: IntentFlags,
    /// 2-4 candidate viewpoints / solutions / theoretical frameworks to investigate.
    #[serde(default)]
    pub candidate_frames: Vec<CandidateFrame>,
    pub perspectives: Vec<Perspective>,
    pub stakeholders: Vec<Stakeholder>,
    pub country_coverage: CountryCoverage,
    /// Mode-aware search strategy metadata.
    #[serde(default)]
    pub search_strategy: SearchStrategy,
    pub search_queries: Vec<SearchQuery>,
    /// 2-3+ possible future directions / scenarios.
    #[serde(default)]
    pub future_directions: Vec<FutureDirection>,
    /// Known information gaps after analysis.
    #[serde(default)]
    pub coverage_gaps: Vec<String>,
}

fn default_analysis_mode() -> String {
    "multi_hypothesis_mode".to_string()
}
fn default_question_type() -> String {
    "open_research".to_string()
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
    /// One of: geopolitics, international_economics, social_issue, regional_economy,
    /// stock_market, academic, engineering, philosophy, literature, arts, psychology,
    /// technology, general.
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
// New structs for dual-mode deep research
// ---------------------------------------------------------------------------

/// Conditional flags that control which optional report sections to produce.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct IntentFlags {
    /// Only true when the topic has clear temporal evolution (geopolitics, policy, academic history).
    #[serde(default)]
    pub needs_temporal_evolution: bool,
    /// Only true when the user explicitly asks about probability / likelihood / risk.
    #[serde(default)]
    pub needs_probability: bool,
    /// Only true when the user's intent is to learn about / study the domain.
    #[serde(default)]
    pub needs_learning_path: bool,
}

/// A candidate viewpoint, solution path, or theoretical framework to investigate.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CandidateFrame {
    /// Short identifier, e.g. "frame_1".
    pub id: String,
    /// Human-readable title, e.g. "Scaling-law optimism toward AGI".
    pub title: String,
    /// `"stakeholder_position"` | `"theory"` | `"solution_path"` | `"interpretation"`
    pub frame_type: String,
    /// One-sentence core claim or hypothesis.
    pub core_claim: String,
    /// Languages to search for this frame (stakeholder_mode: country-local; multi_hypothesis: user+en+zh+de).
    #[serde(default)]
    pub target_languages: Vec<String>,
}

/// Mode-aware search strategy metadata.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SearchStrategy {
    /// Total search budget across all frames (10-100).
    #[serde(default = "default_total_budget")]
    pub total_budget: usize,
    /// Policy governing language/source selection per mode.
    #[serde(default)]
    pub mode_policy: ModePolicy,
    /// Source categories to emphasise (e.g. "academic", "preprint", "policy", "finance", "news").
    #[serde(default)]
    pub source_mix: Vec<String>,
    /// Budget allocated per candidate frame.
    #[serde(default)]
    pub allocated_per_frame: Vec<usize>,
}

fn default_total_budget() -> usize {
    30
}

/// Language/source policy that differs between the two modes.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ModePolicy {
    /// stakeholder_mode: true — search each stakeholder's country in its local language first.
    #[serde(default)]
    pub country_local_first: bool,
    /// multi_hypothesis_mode: fixed language set [user_lang, "en", "zh", "de"].
    #[serde(default)]
    pub fixed_languages: Vec<String>,
}

/// A possible future direction or scenario.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct FutureDirection {
    /// Short scenario name.
    pub scenario: String,
    /// Key driving factors.
    #[serde(default)]
    pub drivers: Vec<String>,
    /// What would trigger this scenario.
    #[serde(default)]
    pub triggers: Vec<String>,
    /// Associated risks.
    #[serde(default)]
    pub risks: Vec<String>,
}

// ---------------------------------------------------------------------------
// LLM-based semantic analysis
// ---------------------------------------------------------------------------

/// Result from LLM semantic analysis — covers the fields that benefit from
/// language understanding rather than rule-based matching.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
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
    // --- New fields for dual-mode deep research ---
    #[serde(default = "default_analysis_mode")]
    analysis_mode: String,
    #[serde(default = "default_question_type")]
    question_type: String,
    #[serde(default)]
    comparison_required: bool,
    #[serde(default)]
    intent_flags: IntentFlags,
    #[serde(default)]
    candidate_frames: Vec<CandidateFrame>,
}

/// System prompt for the semantic analysis sub-call.
const SEMANTIC_ANALYSIS_SYSTEM: &str = "\
You are a research question analyzer. Given a question and optional context, \
produce a JSON object with exactly these fields (no markdown, no explanation, \
pure JSON only):

{
  \"topic_classification\": {
    \"domain\": \"<domain>\",
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
  \"analysis_mode\": \"stakeholder_mode|multi_hypothesis_mode\",
  \"question_type\": \"open_research|feasibility|event_probability|learning_oriented\",
  \"comparison_required\": true/false,
  \"intent_flags\": {
    \"needs_temporal_evolution\": true/false,
    \"needs_probability\": true/false,
    \"needs_learning_path\": true/false
  },
  \"candidate_frames\": [
    {
      \"id\": \"frame_1\",
      \"title\": \"...\",
      \"frame_type\": \"stakeholder_position|theory|solution_path|interpretation\",
      \"core_claim\": \"...\",
      \"target_languages\": [\"en\", \"zh\"]
    }
  ],
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
      \"type\": \"government|corporation|ngo|igo|public|media|investor|regulator\",
      \"interest\": \"...\",
      \"position\": \"See search results\",
      \"primary_language\": \"en|zh|ru|fr|de|es|ar|ja|ko|fa|he|it|nl|id|hi|ur|tr|pt\"
    }
  ]
}

Rules:
- domain must be one of: geopolitics, international_economics, social_issue, regional_economy, \
stock_market, academic, engineering, philosophy, literature, arts, psychology, technology, general
- is_social_issue = true for geopolitics, international_economics, social_issue, regional_economy, stock_market
- analysis_mode: use \"stakeholder_mode\" when the question involves identifiable interest parties \
(countries, companies, investors, regulators, social groups in conflict); \
use \"multi_hypothesis_mode\" when the question is about theories, methods, interpretations, \
or technical/philosophical/artistic exploration
- question_type: \"open_research\" for general exploration, \"feasibility\" for how-to/can-we questions, \
\"event_probability\" when probability/likelihood/risk is asked, \"learning_oriented\" when the user \
wants to study or learn about the domain
- comparison_required: true when the question explicitly asks to compare concepts, systems, works, \
or approaches (e.g. literary comparison, economic system comparison)
- intent_flags: needs_temporal_evolution = true ONLY for topics with clear chronological evolution; \
needs_probability = true ONLY when probability/risk/likelihood is explicitly asked; \
needs_learning_path = true ONLY when the user wants to learn/study the field
- candidate_frames: produce 2-4 distinct viewpoints, solution paths, or theoretical frameworks \
that should be investigated separately. For stakeholder_mode, these are competing positions or \
policy approaches. For multi_hypothesis_mode, these are different theories, methods, or interpretations.
- For stakeholder_mode: target_languages of each frame should use the stakeholder's local language. \
For multi_hypothesis_mode: target_languages should always include the user's language plus en, zh, de.
- Identify ALL relevant countries/stakeholders, even if not explicitly named
- key_points should decompose the question into 2-5 analytical sub-points
- perspectives should include at least 2 distinct viewpoints
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
// Rule-based fallback (Step 1: Topic classification)
//
// This is a MINIMAL fallback used ONLY when LLM semantic analysis fails.
// The LLM is the primary classifier — it handles all 13 domains correctly.
// This fallback returns "general" domain with safe defaults to avoid the
// misclassification issues inherent in keyword-based rules.
// ---------------------------------------------------------------------------

fn classify_topic(_question: &str) -> TopicClassification {
    // When LLM is unavailable, return safe defaults.
    // The LLM prompt (SEMANTIC_ANALYSIS_SYSTEM) is the authoritative classifier
    // for all 13 domains: geopolitics, international_economics, social_issue,
    // regional_economy, stock_market, academic, engineering, philosophy,
    // literature, arts, psychology, technology, general.
    TopicClassification {
        domain: "general".to_string(),
        is_social_issue: false,
        is_controversial: false,
        temporal_relevance: "both".to_string(),
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

/// Top-level query generator — dispatches to the mode-specific branch.
#[allow(clippy::too_many_arguments)]
fn generate_search_queries(
    question: &str,
    key_points: &[KeyPoint],
    stakeholders: &[Stakeholder],
    candidate_frames: &[CandidateFrame],
    sources: &serde_json::Value,
    source_policy: &serde_json::Value,
    depth: &str,
    analysis_mode: &str,
    user_lang: &str,
) -> (Vec<SearchQuery>, SearchStrategy) {
    match analysis_mode {
        "stakeholder_mode" => generate_queries_stakeholder_mode(
            question,
            key_points,
            stakeholders,
            candidate_frames,
            sources,
            source_policy,
            depth,
        ),
        _ => generate_queries_multi_hypothesis_mode(
            question,
            key_points,
            candidate_frames,
            depth,
            user_lang,
        ),
    }
}

/// **stakeholder_mode**: search each stakeholder's country in its local language
/// using the country source registry (existing mechanism preserved).
fn generate_queries_stakeholder_mode(
    question: &str,
    key_points: &[KeyPoint],
    stakeholders: &[Stakeholder],
    candidate_frames: &[CandidateFrame],
    sources: &serde_json::Value,
    source_policy: &serde_json::Value,
    depth: &str,
) -> (Vec<SearchQuery>, SearchStrategy) {
    let mut queries = Vec::new();
    let sources_map = sources.as_object();

    let primary_pool_size = source_policy
        .get("search")
        .and_then(|s| s.get("primary_pool_size"))
        .and_then(|v| v.as_u64())
        .unwrap_or(10) as usize;

    let search_categories = [
        ("official_media", "tier1"),
        ("think_tank", "tier2"),
        ("mainstream_media", "tier3"),
        ("regional_media", "tier4"),
        ("academic", "tier2"),
    ];

    // --- Per-stakeholder country-local queries (preserved from original) ---
    for stakeholder in stakeholders {
        let code = &stakeholder.country_code;
        let lang = &stakeholder.primary_language;

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

        let priority: Vec<String> = country_domains
            .iter()
            .take(primary_pool_size)
            .cloned()
            .collect();

        // Primary query in stakeholder's LOCAL language
        queries.push(SearchQuery {
            query: format_query_for_language(question, lang),
            language: lang.clone(),
            target_country: code.clone(),
            target_category: "mainstream_media".to_string(),
            priority_domains: priority.clone(),
            pool_tier: "top10".to_string(),
        });

        // Category-specific queries for deep/standard
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

        // High-importance key point queries
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

    // --- Per-frame targeted queries (new) ---
    for frame in candidate_frames {
        for lang in &frame.target_languages {
            queries.push(SearchQuery {
                query: format_query_for_language(&frame.core_claim, lang),
                language: lang.clone(),
                target_country: "INTL".to_string(),
                target_category: "think_tank".to_string(),
                priority_domains: vec![],
                pool_tier: "top10".to_string(),
            });
        }
    }

    let total = queries.len();
    let per_frame = allocate_budget_per_frame(total, candidate_frames.len());
    let strategy = SearchStrategy {
        total_budget: total.clamp(10, 100),
        mode_policy: ModePolicy {
            country_local_first: true,
            fixed_languages: vec![],
        },
        source_mix: vec![
            "official_media".to_string(),
            "think_tank".to_string(),
            "mainstream_media".to_string(),
            "policy".to_string(),
            "finance".to_string(),
        ],
        allocated_per_frame: per_frame,
    };

    (queries, strategy)
}

/// **multi_hypothesis_mode**: fixed language set [user_lang, en, zh, de],
/// sources biased toward academic journals, preprints, tech communities, Q&A sites.
fn generate_queries_multi_hypothesis_mode(
    question: &str,
    key_points: &[KeyPoint],
    candidate_frames: &[CandidateFrame],
    depth: &str,
    user_lang: &str,
) -> (Vec<SearchQuery>, SearchStrategy) {
    let mut queries = Vec::new();

    // Fixed language set (deduplicated)
    let mut langs: Vec<String> = vec![
        user_lang.to_string(),
        "en".to_string(),
        "zh".to_string(),
        "de".to_string(),
    ];
    langs.sort();
    langs.dedup();

    // Source categories for multi_hypothesis_mode
    let academic_domains = vec![
        "scholar.google.com".to_string(),
        "arxiv.org".to_string(),
        "semanticscholar.org".to_string(),
        "jstor.org".to_string(),
        "pubmed.ncbi.nlm.nih.gov".to_string(),
    ];
    let preprint_domains = vec![
        "arxiv.org".to_string(),
        "biorxiv.org".to_string(),
        "ssrn.com".to_string(),
        "philpapers.org".to_string(),
    ];
    let community_domains = vec![
        "stackoverflow.com".to_string(),
        "reddit.com".to_string(),
        "news.ycombinator.com".to_string(),
        "github.com".to_string(),
        "zhihu.com".to_string(),
        "quora.com".to_string(),
    ];
    let knowledge_domains = vec![
        "en.wikipedia.org".to_string(),
        "plato.stanford.edu".to_string(),
        "britannica.com".to_string(),
        "de.wikipedia.org".to_string(),
        "zh.wikipedia.org".to_string(),
    ];

    // --- Per-frame + per-language queries ---
    for frame in candidate_frames {
        for lang in &langs {
            // Main frame query
            queries.push(SearchQuery {
                query: format_query_for_language(&frame.core_claim, lang),
                language: lang.clone(),
                target_country: "INTL".to_string(),
                target_category: "academic".to_string(),
                priority_domains: academic_domains.clone(),
                pool_tier: "top10".to_string(),
            });

            // For deep/standard analysis, add preprint and community queries
            if depth == "deep" || depth == "standard" {
                queries.push(SearchQuery {
                    query: format_query_for_language(&frame.title, lang),
                    language: lang.clone(),
                    target_country: "INTL".to_string(),
                    target_category: "preprint".to_string(),
                    priority_domains: preprint_domains.clone(),
                    pool_tier: "top10".to_string(),
                });
            }
        }
    }

    // --- General topic queries in each language ---
    for lang in &langs {
        queries.push(SearchQuery {
            query: format_query_for_language(question, lang),
            language: lang.clone(),
            target_country: "INTL".to_string(),
            target_category: "community".to_string(),
            priority_domains: community_domains.clone(),
            pool_tier: "top10".to_string(),
        });
        queries.push(SearchQuery {
            query: format_query_for_language(question, lang),
            language: lang.clone(),
            target_country: "INTL".to_string(),
            target_category: "knowledge".to_string(),
            priority_domains: knowledge_domains.clone(),
            pool_tier: "top10".to_string(),
        });
    }

    // --- High-importance key point queries ---
    for kp in key_points.iter().filter(|k| k.importance == "high") {
        if kp.point != question && !kp.point.starts_with("Context:") {
            for lang in &langs {
                queries.push(SearchQuery {
                    query: format_query_for_language(&kp.point, lang),
                    language: lang.clone(),
                    target_country: "INTL".to_string(),
                    target_category: "academic".to_string(),
                    priority_domains: academic_domains.clone(),
                    pool_tier: "top10".to_string(),
                });
            }
        }
    }

    let total = queries.len();
    let per_frame = allocate_budget_per_frame(total, candidate_frames.len());
    let strategy = SearchStrategy {
        total_budget: total.clamp(10, 100),
        mode_policy: ModePolicy {
            country_local_first: false,
            fixed_languages: langs,
        },
        source_mix: vec![
            "academic".to_string(),
            "preprint".to_string(),
            "community".to_string(),
            "knowledge".to_string(),
        ],
        allocated_per_frame: per_frame,
    };

    (queries, strategy)
}

/// Distribute total search budget evenly across candidate frames.
fn allocate_budget_per_frame(total_budget: usize, frame_count: usize) -> Vec<usize> {
    if frame_count == 0 {
        return vec![];
    }
    let base = total_budget / frame_count;
    let remainder = total_budget % frame_count;
    let mut alloc: Vec<usize> = vec![base; frame_count];
    // Distribute remainder to the first N frames
    for item in alloc.iter_mut().take(remainder) {
        *item += 1;
    }
    alloc
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
// Candidate frame generation
// ---------------------------------------------------------------------------

/// Build candidate frames for stakeholder_mode.
/// Generates frames based on stakeholders' competing positions or policy approaches.
fn build_candidate_frames_for_stakeholder(
    question: &str,
    stakeholders: &[Stakeholder],
    classification: &TopicClassification,
) -> Vec<CandidateFrame> {
    let mut frames = Vec::new();

    // Generate a frame per major stakeholder (up to 4)
    for (i, s) in stakeholders.iter().take(4).enumerate() {
        frames.push(CandidateFrame {
            id: format!("frame_{}", i + 1),
            title: format!("{} position on the issue", s.name),
            frame_type: "stakeholder_position".to_string(),
            core_claim: format!(
                "{} pursues its interests regarding: {}",
                s.name,
                extract_core_topic(question)
            ),
            target_languages: vec![s.primary_language.clone()],
        });
    }

    // If we have fewer than 2 stakeholders, add generic analytical frames
    if frames.len() < 2 {
        let domain_label = &classification.domain;
        frames.push(CandidateFrame {
            id: format!("frame_{}", frames.len() + 1),
            title: format!("Pro-status-quo perspective on {domain_label}"),
            frame_type: "stakeholder_position".to_string(),
            core_claim: "Arguments favoring the current state of affairs".to_string(),
            target_languages: vec!["en".to_string()],
        });
        frames.push(CandidateFrame {
            id: format!("frame_{}", frames.len() + 1),
            title: format!("Reform / change perspective on {domain_label}"),
            frame_type: "stakeholder_position".to_string(),
            core_claim: "Arguments favoring change or disruption".to_string(),
            target_languages: vec!["en".to_string()],
        });
    }

    frames
}

/// Build candidate frames for multi_hypothesis_mode.
/// Generates frames based on different theories, methods, or interpretations.
fn build_candidate_frames_for_multi_hypothesis(
    question: &str,
    classification: &TopicClassification,
    user_lang: &str,
) -> Vec<CandidateFrame> {
    let mut frames = Vec::new();
    let core = extract_core_topic(question);

    // Fixed language set for multi_hypothesis_mode
    let mut langs: Vec<String> = vec![
        user_lang.to_string(),
        "en".to_string(),
        "zh".to_string(),
        "de".to_string(),
    ];
    langs.dedup();

    let domain = classification.domain.as_str();

    match domain {
        "academic" => {
            frames.push(CandidateFrame {
                id: "frame_1".to_string(),
                title: format!("Mainstream consensus on {core}"),
                frame_type: "theory".to_string(),
                core_claim: "The prevailing academic view supported by peer-reviewed literature"
                    .to_string(),
                target_languages: langs.clone(),
            });
            frames.push(CandidateFrame {
                id: "frame_2".to_string(),
                title: format!("Skeptical / contrarian view on {core}"),
                frame_type: "theory".to_string(),
                core_claim: "Challenges and critiques of the mainstream position".to_string(),
                target_languages: langs.clone(),
            });
            frames.push(CandidateFrame {
                id: "frame_3".to_string(),
                title: format!("Emerging / frontier perspective on {core}"),
                frame_type: "theory".to_string(),
                core_claim: "New research directions that may reshape understanding".to_string(),
                target_languages: langs,
            });
        }
        "engineering" => {
            frames.push(CandidateFrame {
                id: "frame_1".to_string(),
                title: format!("Established approach to {core}"),
                frame_type: "solution_path".to_string(),
                core_claim: "Industry-standard or widely deployed solution".to_string(),
                target_languages: langs.clone(),
            });
            frames.push(CandidateFrame {
                id: "frame_2".to_string(),
                title: format!("Alternative / innovative approach to {core}"),
                frame_type: "solution_path".to_string(),
                core_claim: "Emerging technique or unconventional architecture".to_string(),
                target_languages: langs.clone(),
            });
            frames.push(CandidateFrame {
                id: "frame_3".to_string(),
                title: format!("Hybrid / trade-off analysis for {core}"),
                frame_type: "solution_path".to_string(),
                core_claim: "Combining approaches with cost-benefit trade-off analysis"
                    .to_string(),
                target_languages: langs,
            });
        }
        "philosophy" => {
            frames.push(CandidateFrame {
                id: "frame_1".to_string(),
                title: format!("Classical / traditional view on {core}"),
                frame_type: "interpretation".to_string(),
                core_claim: "Foundational philosophical arguments and their lineage".to_string(),
                target_languages: langs.clone(),
            });
            frames.push(CandidateFrame {
                id: "frame_2".to_string(),
                title: format!("Modern / analytical view on {core}"),
                frame_type: "interpretation".to_string(),
                core_claim: "Contemporary reinterpretation or analytical treatment".to_string(),
                target_languages: langs.clone(),
            });
            frames.push(CandidateFrame {
                id: "frame_3".to_string(),
                title: format!("Eastern / cross-cultural perspective on {core}"),
                frame_type: "interpretation".to_string(),
                core_claim: "Non-Western philosophical traditions and their contributions"
                    .to_string(),
                target_languages: langs,
            });
        }
        "literature" | "arts" => {
            frames.push(CandidateFrame {
                id: "frame_1".to_string(),
                title: format!("Formalist / structural analysis of {core}"),
                frame_type: "interpretation".to_string(),
                core_claim: "Focus on structure, technique, and form".to_string(),
                target_languages: langs.clone(),
            });
            frames.push(CandidateFrame {
                id: "frame_2".to_string(),
                title: format!("Historical-cultural reading of {core}"),
                frame_type: "interpretation".to_string(),
                core_claim: "Interpretation through historical and cultural context".to_string(),
                target_languages: langs.clone(),
            });
            frames.push(CandidateFrame {
                id: "frame_3".to_string(),
                title: format!("Contemporary reception and influence of {core}"),
                frame_type: "interpretation".to_string(),
                core_claim: "Modern reception, adaptations, and ongoing influence".to_string(),
                target_languages: langs,
            });
        }
        "psychology" => {
            frames.push(CandidateFrame {
                id: "frame_1".to_string(),
                title: format!("Cognitive-behavioral perspective on {core}"),
                frame_type: "theory".to_string(),
                core_claim: "Explanation through cognitive processes and behavioral patterns"
                    .to_string(),
                target_languages: langs.clone(),
            });
            frames.push(CandidateFrame {
                id: "frame_2".to_string(),
                title: format!("Psychodynamic / depth-psychology view on {core}"),
                frame_type: "theory".to_string(),
                core_claim: "Explanation through unconscious processes, developmental factors"
                    .to_string(),
                target_languages: langs.clone(),
            });
            frames.push(CandidateFrame {
                id: "frame_3".to_string(),
                title: format!("Neuroscience / biological perspective on {core}"),
                frame_type: "theory".to_string(),
                core_claim: "Explanation through neural mechanisms and biological substrates"
                    .to_string(),
                target_languages: langs,
            });
        }
        "technology" => {
            frames.push(CandidateFrame {
                id: "frame_1".to_string(),
                title: format!("Optimistic / transformative view on {core}"),
                frame_type: "theory".to_string(),
                core_claim: "This technology will fundamentally transform the field".to_string(),
                target_languages: langs.clone(),
            });
            frames.push(CandidateFrame {
                id: "frame_2".to_string(),
                title: format!("Cautious / incremental view on {core}"),
                frame_type: "theory".to_string(),
                core_claim: "Gradual improvement with significant remaining challenges"
                    .to_string(),
                target_languages: langs.clone(),
            });
            frames.push(CandidateFrame {
                id: "frame_3".to_string(),
                title: format!("Critical / risk-focused view on {core}"),
                frame_type: "theory".to_string(),
                core_claim: "Potential dangers, ethical concerns, and unintended consequences"
                    .to_string(),
                target_languages: langs,
            });
        }
        _ => {
            // Generic frames for "general" or unmatched domains
            frames.push(CandidateFrame {
                id: "frame_1".to_string(),
                title: format!("Mainstream view on {core}"),
                frame_type: "theory".to_string(),
                core_claim: "The most widely held position".to_string(),
                target_languages: langs.clone(),
            });
            frames.push(CandidateFrame {
                id: "frame_2".to_string(),
                title: format!("Alternative view on {core}"),
                frame_type: "theory".to_string(),
                core_claim: "A significant dissenting or complementary perspective".to_string(),
                target_languages: langs,
            });
        }
    }

    frames
}

/// Ensure at least `min` candidate frames exist; pad with generic frames if needed.
fn ensure_min_frames(mut frames: Vec<CandidateFrame>, min: usize) -> Vec<CandidateFrame> {
    let mut next_id = frames.len() + 1;
    while frames.len() < min {
        frames.push(CandidateFrame {
            id: format!("frame_{next_id}"),
            title: format!("Additional perspective #{next_id}"),
            frame_type: "theory".to_string(),
            core_claim: "An additional viewpoint requiring investigation".to_string(),
            target_languages: vec!["en".to_string()],
        });
        next_id += 1;
    }
    frames
}

// ---------------------------------------------------------------------------
// Mode routing & intent detection
// ---------------------------------------------------------------------------

/// Fallback mode selector — used ONLY when LLM does not provide `analysis_mode`.
/// When the domain is known (from LLM), routes to the correct mode.
/// When unknown (fallback), defaults to `multi_hypothesis_mode` which is safer
/// than incorrectly triggering stakeholder-based country-local search.
fn select_analysis_mode(classification: &TopicClassification, _question: &str) -> String {
    // Domains that map to stakeholder_mode
    if matches!(
        classification.domain.as_str(),
        "geopolitics"
            | "international_economics"
            | "social_issue"
            | "regional_economy"
            | "stock_market"
    ) {
        return "stakeholder_mode".to_string();
    }

    // Everything else (including "general") → multi_hypothesis_mode
    "multi_hypothesis_mode".to_string()
}

/// Fallback intent flags — used ONLY when LLM does not provide `intent_flags`.
/// All flags default to false; the LLM is the primary detector.
fn detect_intent_flags(
    _question: &str,
    _intent: &UserIntent,
    _classification: &TopicClassification,
) -> IntentFlags {
    IntentFlags::default()
}

/// Fallback question type — used ONLY when LLM does not provide `question_type`.
fn infer_question_type(_question: &str, _intent: &UserIntent) -> String {
    "open_research".to_string()
}

/// Fallback comparison flag — used ONLY when LLM does not provide `comparison_required`.
fn detect_comparison_required(_question: &str) -> bool {
    false
}

/// Derive user's language from question text heuristics.
fn derive_user_language(question: &str) -> String {
    // Check for significant CJK character presence
    let cjk_count = question
        .chars()
        .filter(|c| ('\u{4E00}'..='\u{9FFF}').contains(c))
        .count();
    let total_chars = question.chars().count().max(1);
    if cjk_count * 100 / total_chars > 20 {
        return "zh".to_string();
    }

    // Check for Japanese-specific characters (hiragana/katakana)
    if question
        .chars()
        .any(|c| ('\u{3040}'..='\u{30FF}').contains(&c))
    {
        return "ja".to_string();
    }

    // Check for Korean-specific characters (hangul)
    if question
        .chars()
        .any(|c| ('\u{AC00}'..='\u{D7AF}').contains(&c))
    {
        return "ko".to_string();
    }

    // Check for Arabic script
    if question
        .chars()
        .any(|c| ('\u{0600}'..='\u{06FF}').contains(&c))
    {
        return "ar".to_string();
    }

    // Check for Cyrillic
    if question
        .chars()
        .any(|c| ('\u{0400}'..='\u{04FF}').contains(&c))
    {
        return "ru".to_string();
    }

    // Check for Devanagari (Hindi)
    if question
        .chars()
        .any(|c| ('\u{0900}'..='\u{097F}').contains(&c))
    {
        return "hi".to_string();
    }

    // Default to English
    "en".to_string()
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
    fn test_classify_topic_fallback_always_general() {
        // Rule-based classify_topic is now a minimal fallback — always returns "general".
        // Real classification is done by LLM via SEMANTIC_ANALYSIS_SYSTEM prompt.
        let c = classify_topic("What are the geopolitical implications of NATO expansion?");
        assert_eq!(c.domain, "general");
        assert!(!c.is_social_issue);

        let c2 = classify_topic("中美芯片战争对全球半导体供应链的影响");
        assert_eq!(c2.domain, "general");

        let c3 = classify_topic("存在主义哲学如何看待自由意志问题？");
        assert_eq!(c3.domain, "general");
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
        // Without LLM, domain falls back to "general", mode to "multi_hypothesis_mode"
        assert_eq!(parsed.topic_classification.domain, "general");
        assert_eq!(parsed.analysis_mode, "multi_hypothesis_mode");
        // Country detection and stakeholders still work (rule-based)
        assert!(!parsed.stakeholders.is_empty());
        assert!(parsed.stakeholders.iter().any(|s| s.country_code == "CN"));
        assert!(parsed.stakeholders.iter().any(|s| s.country_code == "US"));
        // Structural integrity: queries, frames, future directions present
        assert!(!parsed.search_queries.is_empty());
        assert!(parsed.candidate_frames.len() >= 2);
        assert!(parsed.future_directions.len() >= 2);
        // Intent flags default to all-false without LLM
        assert!(!parsed.intent_flags.needs_temporal_evolution);
        assert!(!parsed.intent_flags.needs_probability);
        assert!(!parsed.intent_flags.needs_learning_path);
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

    // ═══════════════════════════════════════════════════════════
    // Dual-mode deep research tests
    // ═══════════════════════════════════════════════════════════

    // --- Mode routing: when LLM provides a domain, routing works correctly ---

    #[test]
    fn test_mode_routing_stakeholder_domains() {
        // All stakeholder domains route to stakeholder_mode
        for domain in &[
            "geopolitics",
            "international_economics",
            "social_issue",
            "regional_economy",
            "stock_market",
        ] {
            let c = TopicClassification {
                domain: domain.to_string(),
                is_social_issue: true,
                ..Default::default()
            };
            assert_eq!(
                select_analysis_mode(&c, ""),
                "stakeholder_mode",
                "domain '{domain}' should route to stakeholder_mode"
            );
        }
    }

    #[test]
    fn test_mode_routing_multi_hypothesis_domains() {
        // All multi_hypothesis domains route to multi_hypothesis_mode
        for domain in &[
            "academic",
            "engineering",
            "philosophy",
            "literature",
            "arts",
            "psychology",
            "technology",
        ] {
            let c = TopicClassification {
                domain: domain.to_string(),
                ..Default::default()
            };
            assert_eq!(
                select_analysis_mode(&c, ""),
                "multi_hypothesis_mode",
                "domain '{domain}' should route to multi_hypothesis_mode"
            );
        }
    }

    #[test]
    fn test_mode_routing_fallback_defaults_to_multi_hypothesis() {
        // When LLM is unavailable, classify_topic returns "general",
        // which safely defaults to multi_hypothesis_mode.
        let c = classify_topic("any question at all");
        assert_eq!(c.domain, "general");
        assert_eq!(select_analysis_mode(&c, ""), "multi_hypothesis_mode");
    }

    // --- Intent flags: fallback returns all-false (safe default) ---

    #[test]
    fn test_intent_flags_fallback_all_false() {
        // When LLM is unavailable, all flags default to false.
        let intent = UserIntent::default();
        let c = TopicClassification::default();
        let flags = detect_intent_flags("any question", &intent, &c);
        assert!(!flags.needs_temporal_evolution);
        assert!(!flags.needs_probability);
        assert!(!flags.needs_learning_path);
    }

    // --- LLM output takes priority over fallback ---

    #[test]
    fn test_llm_semantic_result_fields_used() {
        // Simulate LLM returning full semantic result
        let llm = LlmSemanticResult {
            topic_classification: TopicClassification {
                domain: "philosophy".to_string(),
                is_social_issue: false,
                is_controversial: false,
                temporal_relevance: "both".to_string(),
            },
            analysis_mode: "multi_hypothesis_mode".to_string(),
            question_type: "open_research".to_string(),
            comparison_required: false,
            intent_flags: IntentFlags {
                needs_temporal_evolution: true,
                needs_probability: false,
                needs_learning_path: true,
            },
            candidate_frames: vec![
                CandidateFrame {
                    id: "frame_1".to_string(),
                    title: "Classical view".to_string(),
                    frame_type: "interpretation".to_string(),
                    core_claim: "Traditional philosophical arguments".to_string(),
                    target_languages: vec!["en".to_string(), "de".to_string()],
                },
                CandidateFrame {
                    id: "frame_2".to_string(),
                    title: "Modern view".to_string(),
                    frame_type: "interpretation".to_string(),
                    core_claim: "Contemporary reinterpretation".to_string(),
                    target_languages: vec!["en".to_string(), "zh".to_string()],
                },
            ],
            ..Default::default()
        };

        // Verify LLM fields are directly usable
        assert_eq!(llm.topic_classification.domain, "philosophy");
        assert_eq!(llm.analysis_mode, "multi_hypothesis_mode");
        assert!(llm.intent_flags.needs_temporal_evolution);
        assert!(llm.intent_flags.needs_learning_path);
        assert!(!llm.intent_flags.needs_probability);
        assert_eq!(llm.candidate_frames.len(), 2);
    }

    #[test]
    fn test_candidate_frames_minimum_two() {
        let frames = ensure_min_frames(vec![], 2);
        assert!(frames.len() >= 2);

        let single = vec![CandidateFrame {
            id: "frame_1".to_string(),
            title: "Only one".to_string(),
            ..Default::default()
        }];
        let padded = ensure_min_frames(single, 2);
        assert!(padded.len() >= 2);
    }

    #[test]
    fn test_query_generation_stakeholder_country_local_first() {
        let stakeholders = vec![
            Stakeholder {
                name: "China".to_string(),
                country_code: "CN".to_string(),
                stakeholder_type: "government".to_string(),
                interest: "test".to_string(),
                position: "test".to_string(),
                primary_language: "zh".to_string(),
            },
            Stakeholder {
                name: "Japan".to_string(),
                country_code: "JP".to_string(),
                stakeholder_type: "government".to_string(),
                interest: "test".to_string(),
                position: "test".to_string(),
                primary_language: "ja".to_string(),
            },
        ];
        let frames = vec![CandidateFrame {
            id: "f1".to_string(),
            title: "test".to_string(),
            frame_type: "stakeholder_position".to_string(),
            core_claim: "test claim".to_string(),
            target_languages: vec!["zh".to_string()],
        }];
        let sources = serde_json::json!({
            "CN": { "lang": "zh", "official_media": ["xinhuanet.com"], "think_tank": [], "mainstream_media": [], "regional_media": [], "academic": [] },
            "JP": { "lang": "ja", "official_media": ["kantei.go.jp"], "think_tank": [], "mainstream_media": [], "regional_media": [], "academic": [] }
        });
        let policy = serde_json::json!({ "search": { "primary_pool_size": 5 } });

        let (queries, strategy) = generate_queries_stakeholder_mode(
            "test question",
            &[],
            &stakeholders,
            &frames,
            &sources,
            &policy,
            "standard",
        );

        assert!(strategy.mode_policy.country_local_first);
        // Should have queries in zh and ja (local languages)
        assert!(queries.iter().any(|q| q.language == "zh"));
        assert!(queries.iter().any(|q| q.language == "ja"));
    }

    #[test]
    fn test_query_generation_multi_hypothesis_fixed_languages() {
        let frames = vec![
            CandidateFrame {
                id: "f1".to_string(),
                title: "Mainstream view".to_string(),
                frame_type: "theory".to_string(),
                core_claim: "LLMs can achieve AGI".to_string(),
                target_languages: vec![
                    "en".to_string(),
                    "zh".to_string(),
                    "de".to_string(),
                ],
            },
            CandidateFrame {
                id: "f2".to_string(),
                title: "Skeptical view".to_string(),
                frame_type: "theory".to_string(),
                core_claim: "LLMs cannot achieve AGI".to_string(),
                target_languages: vec![
                    "en".to_string(),
                    "zh".to_string(),
                    "de".to_string(),
                ],
            },
        ];

        let (queries, strategy) = generate_queries_multi_hypothesis_mode(
            "Can LLMs achieve AGI?",
            &[],
            &frames,
            "standard",
            "en",
        );

        assert!(!strategy.mode_policy.country_local_first);
        // Fixed languages should include en, zh, de
        assert!(strategy.mode_policy.fixed_languages.contains(&"en".to_string()));
        assert!(strategy.mode_policy.fixed_languages.contains(&"zh".to_string()));
        assert!(strategy.mode_policy.fixed_languages.contains(&"de".to_string()));
        // Queries should exist in multiple languages
        assert!(queries.iter().any(|q| q.language == "en"));
        assert!(queries.iter().any(|q| q.language == "zh"));
        assert!(queries.iter().any(|q| q.language == "de"));
    }

    #[test]
    fn test_search_budget_within_10_100() {
        let frames = vec![
            CandidateFrame {
                id: "f1".to_string(),
                title: "View A".to_string(),
                frame_type: "theory".to_string(),
                core_claim: "Claim A".to_string(),
                target_languages: vec!["en".to_string()],
            },
            CandidateFrame {
                id: "f2".to_string(),
                title: "View B".to_string(),
                frame_type: "theory".to_string(),
                core_claim: "Claim B".to_string(),
                target_languages: vec!["en".to_string()],
            },
        ];

        let (_, strategy) = generate_queries_multi_hypothesis_mode(
            "Test question about physics",
            &[],
            &frames,
            "standard",
            "en",
        );

        assert!(strategy.total_budget >= 10);
        assert!(strategy.total_budget <= 100);
    }

    #[test]
    fn test_comparison_required_fallback() {
        // Fallback always returns false; LLM handles comparison detection.
        assert!(!detect_comparison_required("Compare Python vs Rust"));
        assert!(!detect_comparison_required("What is quantum computing?"));
    }

    #[test]
    fn test_question_type_fallback() {
        // Fallback always returns "open_research"; LLM handles type inference.
        let intent = UserIntent::default();
        assert_eq!(
            infer_question_type("What is the probability of a recession?", &intent),
            "open_research"
        );
        assert_eq!(
            infer_question_type("How to optimize LLMs for mobile devices?", &intent),
            "open_research"
        );
    }

    #[test]
    fn test_derive_user_language() {
        assert_eq!(derive_user_language("中美芯片战争的影响"), "zh");
        assert_eq!(
            derive_user_language("What is the impact of trade wars?"),
            "en"
        );
        assert_eq!(derive_user_language("量子コンピューティングとは何ですか"), "ja");
        assert_eq!(derive_user_language("양자 컴퓨팅이란 무엇입니까"), "ko");
    }

    #[test]
    fn test_allocate_budget_per_frame() {
        assert_eq!(allocate_budget_per_frame(30, 3), vec![10, 10, 10]);
        assert_eq!(allocate_budget_per_frame(31, 3), vec![11, 10, 10]);
        assert_eq!(allocate_budget_per_frame(10, 0), Vec::<usize>::new());
    }
}
