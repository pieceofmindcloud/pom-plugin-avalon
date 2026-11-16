//! The engine settings this plugin offers to a deployment.
//!
//! The POM knows no engine: an inference service publishes the settings it
//! takes (`host.services`, field `settings`), the deploy inspector draws
//! them, and the chosen values come back in `engine.load`
//! (`params.settings`). `launch::command` maps them to the engine's flags.
//! Publish only settings the engine really reads, with the engine's own
//! limits; `placement: "pom"` puts one in the POM settings card instead of
//! the engine's.

use serde_json::{json, Map, Value};

pub struct Setting {
    pub key: &'static str,
    /// `number`, `integer`, `boolean` or `select`.
    pub kind: &'static str,
    pub en: (&'static str, &'static str),
    pub pt: (&'static str, &'static str),
    pub min: Option<f64>,
    pub max: Option<f64>,
    pub options: &'static [&'static str],
    pub pom: bool,
}

/// The engine's settings: `(label, description)` in English and Portuguese.
pub fn all() -> Vec<Setting> {
    vec![
        Setting {
            key: "threads",
            kind: "integer",
            en: (
                "CPU threads",
                "Threads that compute the experts missing from the GPU (0 = the engine's default).",
            ),
            pt: (
                "Threads de CPU",
                "Threads que calculam os experts que faltam na GPU (0 = o padrão do motor).",
            ),
            min: Some(0.0),
            max: Some(256.0),
            options: &[],
            pom: false,
        },
        Setting {
            key: "mtp",
            kind: "boolean",
            en: (
                "MTP speculative decoding",
                "Drafts tokens with the MTP head found in the model's MTP folder (about 45% faster generation).",
            ),
            pt: (
                "Decodificação especulativa MTP",
                "Rascunha tokens com a cabeça MTP da pasta MTP do modelo (geração cerca de 45% mais rápida).",
            ),
            min: None,
            max: None,
            options: &[],
            pom: false,
        },
        Setting {
            key: "draft_n",
            kind: "integer",
            en: ("MTP draft tokens", "Tokens the MTP head drafts per step (Avalon measures 2 as best)."),
            pt: ("Tokens de rascunho MTP", "Tokens que a cabeça MTP rascunha por passo (o Avalon mede 2 como o melhor)."),
            min: Some(1.0),
            max: Some(8.0),
            options: &[],
            pom: false,
        },
        Setting {
            key: "flash_attn",
            kind: "boolean",
            en: ("Flash attention", "On by default (Avalon's choice). Off only as a fallback for a card where it misbehaves; without it the V cache stays f16."),
            pt: ("Flash attention", "Ligado por padrão (escolha do Avalon). Desligue só como alternativa numa placa em que ele falhe; sem ele o cache V fica em f16."),
            min: None,
            max: None,
            options: &[],
            pom: false,
        },
        Setting {
            key: "expert_ram_gb",
            kind: "integer",
            en: ("RAM for experts (GB)", "Host RAM this machine gives the experts the GPU does not hold (0 = automatic: every expert copied to RAM when they all fit, else a smaller arena). Set it when several stages share one machine: each measures the free RAM alone."),
            pt: ("RAM para experts (GB)", "RAM que esta máquina dá aos experts que a GPU não guarda (0 = automático: todos copiados para a RAM quando cabem, senão uma arena menor). Defina quando vários estágios dividem uma máquina: cada um mede a RAM livre sozinho."),
            min: Some(0.0),
            max: Some(4096.0),
            options: &[],
            pom: false,
        },
        Setting {
            key: "tail_skip",
            kind: "boolean",
            en: ("Skip rare missed experts", "Experts every token of a verification routes only at rank 7 or lower are skipped when missing from the GPU (Avalon's default, same GSM8K score)."),
            pt: ("Pular experts raros ausentes", "Experts que todo token de uma verificação roteia só no posto 7 ou abaixo são pulados quando faltam na GPU (padrão do Avalon, mesmo GSM8K)."),
            min: None,
            max: None,
            options: &[],
            pom: false,
        },
        Setting {
            key: "gpu_stages",
            kind: "boolean",
            en: ("Pipeline over two GPUs", "With two cards visible, the layers run as a two-stage pipeline instead of a plain layer split (Avalon's default)."),
            pt: ("Pipeline em duas GPUs", "Com duas placas visíveis, as camadas rodam como um pipeline de dois estágios em vez de uma divisão simples (padrão do Avalon)."),
            min: None,
            max: None,
            options: &[],
            pom: false,
        },
    ]
}

pub fn schema() -> Value {
    Value::Array(
        all()
            .into_iter()
            .map(|setting| {
                let mut entry = json!({
                    "key": setting.key,
                    "type": setting.kind,
                    "label": {"en": setting.en.0, "pt-BR": setting.pt.0},
                    "description": {"en": setting.en.1, "pt-BR": setting.pt.1},
                });
                if let Some(min) = setting.min {
                    entry["min"] = json!(min);
                }
                if let Some(max) = setting.max {
                    entry["max"] = json!(max);
                }
                if !setting.options.is_empty() {
                    entry["options"] = json!(setting.options);
                }
                if setting.pom {
                    entry["placement"] = json!("pom");
                }
                entry
            })
            .collect(),
    )
}

/// The values of a load request that name a published key and have its type;
/// the rest are dropped, never passed to the engine.
pub fn select(values: &Value) -> Map<String, Value> {
    let Some(values) = values.as_object() else {
        return Map::new();
    };
    all()
        .into_iter()
        .filter_map(|setting| {
            let value = values.get(setting.key)?;
            let in_range = |number: f64| {
                setting.min.is_none_or(|min| number >= min)
                    && setting.max.is_none_or(|max| number <= max)
            };
            let valid = match setting.kind {
                "integer" => value.as_i64().is_some_and(|number| in_range(number as f64)),
                "number" => value.as_f64().is_some_and(in_range),
                "boolean" => value.is_boolean(),
                "select" => value
                    .as_str()
                    .is_some_and(|choice| setting.options.contains(&choice)),
                _ => false,
            };
            valid.then(|| (setting.key.to_owned(), value.clone()))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_setting_has_both_languages_and_a_known_type() {
        for setting in all() {
            assert!(["number", "integer", "boolean", "select"].contains(&setting.kind));
            assert!(!setting.en.0.is_empty() && !setting.pt.0.is_empty());
            assert_eq!(setting.kind == "select", !setting.options.is_empty());
        }
    }

    #[test]
    fn only_published_and_valid_values_reach_the_engine() {
        let chosen = select(&json!({"threads": 8, "unknown": 1}));
        assert_eq!(chosen.get("threads"), Some(&json!(8)));
        assert!(chosen.get("unknown").is_none());
        assert!(select(&json!({"threads": -1})).is_empty());
        assert!(select(&json!({"threads": "8"})).is_empty());
        assert!(select(&Value::Null).is_empty());
    }
}
