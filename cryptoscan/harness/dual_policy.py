"""Dual-policy: run cheap rule first; only escalate to LLM when rule says
long/short. Final decision is the consensus, with conf reduced and a clear
'rule_vs_llm' annotation stored in tags for later analysis."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .agent import Decision, rule_policy


@dataclass
class DualPolicy:
    """Compose rule_policy + an LLM policy. Use cases:

    rule == watch/skip                  -> return rule decision (no LLM call)
    rule == long/short, LLM agrees      -> consensus, keep direction, average conf
    rule == long/short, LLM disagrees   -> downgrade to 'watch' with min(conf)
    """

    llm_policy: Any  # callable taking snapshot -> Decision
    tools_called: list[dict[str, Any]] | None = None  # mirrors LLMPolicy attr for persistence
    similar_episode_ids: list[str] | None = None      # mirrors LLMPolicy attr for persistence

    # Thresholds for "interesting enough to spend LLM tokens even when rule=watch"
    escalate_min_abs_oi_pct: float = 3.0
    escalate_min_volume_usdt: float = 1_000_000

    def _is_interesting(self, snapshot: dict[str, Any]) -> bool:
        oi = abs(float(snapshot.get("oi_change_pct") or 0))
        fr = float(snapshot.get("curr_funding") or 0)
        vol = float(snapshot.get("volume_24h") or 0)
        # negative funding + non-trivial OI move + liquid pair → worth asking LLM
        return oi >= self.escalate_min_abs_oi_pct and fr < 0 and vol >= self.escalate_min_volume_usdt

    def __call__(self, snapshot: dict[str, Any]) -> Decision:
        rule = rule_policy(snapshot)

        # Cheapest path: rule=skip is an explicit veto, never escalate.
        if rule.decision == "skip":
            self.tools_called = []
            self.similar_episode_ids = []
            tags = list(rule.tags) + ["dual:rule_only", "rule:skip"]
            return Decision(
                decision="skip",
                confidence=rule.confidence,
                rationale="[rule-only] " + rule.rationale,
                entry_plan=rule.entry_plan,
                tags=tags,
            )

        # rule=watch: only escalate if the snapshot is interesting; otherwise cheap-out.
        if rule.decision == "watch" and not self._is_interesting(snapshot):
            self.tools_called = []
            self.similar_episode_ids = []
            tags = list(rule.tags) + ["dual:rule_only", "rule:watch"]
            return Decision(
                decision="watch",
                confidence=rule.confidence,
                rationale="[rule-only] " + rule.rationale,
                entry_plan=rule.entry_plan,
                tags=tags,
            )

        # Escalate to LLM (rule is long/short, OR rule=watch but signals are interesting)
        llm = self.llm_policy(snapshot)
        self.tools_called = list(getattr(self.llm_policy, "tools_called", None) or [])
        self.similar_episode_ids = list(getattr(self.llm_policy, "similar_episode_ids", None) or [])

        agree = rule.decision == llm.decision
        rule_actionable = rule.decision in {"long", "short"}
        llm_actionable = llm.decision in {"long", "short"}

        if agree and rule_actionable:
            # Strong consensus: both sides actionable and agree on direction.
            conf = (rule.confidence + llm.confidence) / 2
            plan = llm.entry_plan or rule.entry_plan
            tag = f"dual:agree:{rule.decision}"
            merged_tags = list(dict.fromkeys(list(rule.tags) + list(llm.tags) + [tag]))
            rationale = (
                f"[CONSENSUS {rule.decision}] rule:{rule.confidence:.2f}+llm:{llm.confidence:.2f}\n"
                f"  rule: {rule.rationale}\n  llm:  {llm.rationale}"
            )
            return Decision(decision=rule.decision, confidence=round(conf, 3),
                            rationale=rationale, entry_plan=plan, tags=merged_tags)

        if rule.decision == "watch" and llm_actionable:
            # rule was non-committal but signals were interesting; LLM stepped up.
            # Slightly discount confidence vs full consensus.
            conf = max(0.0, llm.confidence - 0.1)
            tag = f"dual:llm_lead:{llm.decision}"
            merged_tags = list(dict.fromkeys(list(rule.tags) + list(llm.tags) + [tag]))
            rationale = (
                f"[LLM-LEAD {llm.decision}] rule=watch({rule.confidence:.2f}) escalated; "
                f"llm={llm.decision}({llm.confidence:.2f})\n"
                f"  rule: {rule.rationale}\n  llm:  {llm.rationale}"
            )
            return Decision(decision=llm.decision, confidence=round(conf, 3),
                            rationale=rationale, entry_plan=llm.entry_plan or rule.entry_plan,
                            tags=merged_tags)

        # All other cases (true disagreement, or LLM also says watch) → conservative watch.
        tag = f"dual:disagree:rule={rule.decision}/llm={llm.decision}"
        merged_tags = list(dict.fromkeys(list(rule.tags) + list(llm.tags) + [tag]))
        rationale = (
            f"[DISAGREE] rule={rule.decision}({rule.confidence:.2f}) vs "
            f"llm={llm.decision}({llm.confidence:.2f}). Downgrading to watch.\n"
            f"  rule: {rule.rationale}\n  llm:  {llm.rationale}"
        )
        return Decision(decision="watch", confidence=min(rule.confidence, llm.confidence),
                        rationale=rationale, entry_plan=llm.entry_plan or rule.entry_plan,
                        tags=merged_tags)
