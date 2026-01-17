The PIPELINE:

🙏🙏🙏🙏🙏🙏🙏🙏🙏🙏🙏🙏🙏🙏🙏🙏🙏🙏🙏🙏🙏🙏🙏

🙏 DEDUP (near) → cleaning → cluster labeling → doc-form definitions → OCR policy per form (tego nie zrobilem w koncu jeszcze) → semantic clustering → targeted image OCR (tego tez nie) 🙏

🙏🙏🙏🙏🙏🙏🙏🙏🙏🙏🙏🙏🙏🙏🙏🙏🙏🙏🙏🙏🙏🙏🙏

Manifest trajectory (order of preprocessing):
manifest_all.csv
->
manifest_with_docform_clusters.csv
->
manifest_dedup.csv
->
manifest_with_doc_types.csv
->
manifest_with_near_dedup.csv
->



Objects
Document: doc_id, source_path, doc_type, text_clean, hash, quality
Mention: doc_id, span_start, span_end, label, text, confidence
Entity: entity_id, entity_type, canonical_name, aliases[]
Event: event_id, event_type, time_start/end/precision, place_entity_id?
Claim (reified edge):
claim_id, subj_id, predicate, obj_id, qualifiers(json), doc_id, span(s), method, confidence

Key predicates (start with these)
MENTIONS (Doc→Mention)
REFERS_TO (Mention→Entity)
PARTICIPATED_IN (Entity→Event) with role qualifiers
LOCATED_AT (Entity/Event→Place)
HAS_ATTRIBUTE (Entity→attribute objects if needed)
ASSERTS (Doc→Claim)