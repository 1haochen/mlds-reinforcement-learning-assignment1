"""Generate the Canvas-ready Assignment 1 PDF from recorded experiment artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Image,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


REPO_URL = "https://github.com/1haochen/mlds-reinforcement-learning-assignment1"


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


def _page(canvas, document) -> None:
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#555555"))
    canvas.drawString(0.65 * inch, 0.42 * inch, "MLDS Assignment 1 - Reinforcement Learning")
    canvas.drawRightString(7.85 * inch, 0.42 * inch, f"Page {document.page}")
    canvas.restoreState()


def _figure(path: Path, caption: str, styles, width: float = 6.65 * inch):
    if not path.exists():
        return [Paragraph(f"Figure unavailable: {path}", styles["Note"])]
    image = Image(str(path))
    image_width, image_height = image.imageWidth, image.imageHeight
    image.drawWidth = width
    image.drawHeight = width * image_height / image_width
    return [KeepTogether([image, Spacer(1, 4), Paragraph(caption, styles["Caption"])])]


def _metrics_table(rows, col_widths=None):
    table = Table(rows, colWidths=col_widths, repeatRows=1, hAlign="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f4e78")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTNAME", (0, 1), (0, -1), "Helvetica-Bold"),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#aab4be")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f2f5f7")]),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    return table


def build_report(artifacts: Path, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    shuttle = _read_json(artifacts / "shuttle/summary.json")
    cart_basic = _read_json(artifacts / "cartpole/basic/summary.json")
    cart_base = _read_json(artifacts / "cartpole/baseline/summary.json")
    pong_basic = _read_json(artifacts / "pong/basic/summary.json")
    pong_base = _read_json(artifacts / "pong/baseline/summary.json")

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="Title2", parent=styles["Title"], fontName="Helvetica-Bold", fontSize=23, leading=28, textColor=colors.HexColor("#17365d"), alignment=TA_CENTER, spaceAfter=16))
    styles.add(ParagraphStyle(name="Subtitle", parent=styles["Normal"], fontSize=12, leading=16, alignment=TA_CENTER, textColor=colors.HexColor("#4d5966"), spaceAfter=18))
    styles.add(ParagraphStyle(name="H1x", parent=styles["Heading1"], fontSize=17, leading=21, textColor=colors.HexColor("#17365d"), spaceBefore=8, spaceAfter=9))
    styles.add(ParagraphStyle(name="H2x", parent=styles["Heading2"], fontSize=13, leading=16, textColor=colors.HexColor("#1f4e78"), spaceBefore=8, spaceAfter=6))
    styles.add(ParagraphStyle(name="Body", parent=styles["BodyText"], fontSize=9.6, leading=13.2, spaceAfter=7))
    styles.add(ParagraphStyle(name="BulletX", parent=styles["Body"], leftIndent=14, firstLineIndent=-7, bulletIndent=5, spaceAfter=4))
    styles.add(ParagraphStyle(name="Caption", parent=styles["Body"], fontSize=8.5, leading=11, alignment=TA_CENTER, textColor=colors.HexColor("#4d5966"), spaceAfter=9))
    styles.add(ParagraphStyle(name="Note", parent=styles["Body"], fontSize=8.8, leading=12, backColor=colors.HexColor("#fff4d6"), borderColor=colors.HexColor("#d9a441"), borderWidth=0.6, borderPadding=7, spaceAfter=8))
    styles.add(ParagraphStyle(name="CodeX", parent=styles["Code"], fontName="Courier", fontSize=7.7, leading=10, backColor=colors.HexColor("#f3f5f7"), borderPadding=6, spaceAfter=8))

    doc = SimpleDocTemplate(str(output), pagesize=letter, rightMargin=0.65 * inch, leftMargin=0.65 * inch, topMargin=0.58 * inch, bottomMargin=0.62 * inch, title="MLDS Assignment 1", author="Yihao Chen")
    story = []
    story.extend([
        Spacer(1, 0.8 * inch),
        Paragraph("Assignment 1", styles["Title2"]),
        Paragraph("Dynamic Programming and Policy Gradients", styles["Subtitle"]),
        Spacer(1, 0.25 * inch),
        _metrics_table([
            ["Submission field", "Value"],
            ["GitHub username", "1haochen"],
            ["Repository", "mlds-reinforcement-learning-assignment1"],
            ["Repository link", Paragraph(f'<link href="{REPO_URL}">Open GitHub repository</link>', styles["Body"])],
            ["Required environments", "CartPole-v1; Pong-v5 (Gymnasium registration ALE/Pong-v5)"],
        ], [1.45 * inch, 5.0 * inch]),
        Spacer(1, 0.22 * inch),
        Paragraph("Executive summary", styles["H1x"]),
        Paragraph("I modeled shuttle dispatch as a finite-state discounted-cost MDP and solved it independently by 500-stage backward induction, value iteration, and policy iteration. All three give the same initial dispatch threshold of 13 customers. I also implemented episodic REINFORCE in PyTorch for CartPole and Pong, then added a learned state-value baseline. The repository preserves raw rewards, configurations, checkpoints, plots, and numerical validation statistics.", styles["Body"]),
        Paragraph("Evidence status", styles["H2x"]),
        Paragraph("The checked-in CartPole runs include the required 500 complete evaluation rollouts. The checked-in Pong figures are end-to-end smoke evidence (5 training episodes and 20 evaluations truncated at 500 steps), not a claim of converged Atari performance. The full 10,000-training/500-evaluation commands are documented and runnable; reporting the smoke status explicitly avoids presenting unperformed computation as experimental fact.", styles["Note"]),
        Paragraph("How to reproduce", styles["H2x"]),
        Paragraph("python -m pip install -e '.[test,atari]'<br/>python -m mlds_rl.cli --output artifacts shuttle<br/>python -m mlds_rl.cli --output artifacts cartpole --episodes 1500<br/>python -m mlds_rl.cli --output artifacts cartpole --baseline --episodes 1500<br/>python -m mlds_rl.cli --output artifacts pong --episodes 10000<br/>python -m mlds_rl.cli --output artifacts pong --baseline --episodes 10000<br/>PYTHONPATH=src pytest -q", styles["CodeX"]),
        PageBreak(),
        Paragraph("1. Shuttle dispatch: model and methods", styles["H1x"]),
        Paragraph("State s is the number of waiting passengers in {0,...,200}; action a=0 waits and a=1 dispatches one shuttle. I assume dispatch occurs before the next arrival. Let q=max(0,s-15a), A be uniform on {1,...,5}, and s'=min(200,q+A). The transition probability P(s'|s,a) is obtained by enumeration. The one-step expected cost is E[100a+2s']. This makes the boundary behavior explicit: arrivals that would exceed capacity accumulate at state 200. The objective minimizes expected discounted cost with gamma=0.95.", styles["Body"]),
        Paragraph("For finite T=500, backward induction sets V_500(s)=0 and computes V_t(s)=min_a[c(s,a)+gamma sum P(s'|s,a)V_(t+1)(s')]. Value iteration applies the same Bellman operator to convergence for T=infinity. Policy iteration alternates exact linear-system evaluation with greedy improvement.", styles["Body"]),
        Paragraph("Quantitative validation", styles["H2x"]),
        _metrics_table([
            ["Check", "Recorded result"],
            ["Transition row-sum maximum error", f"{shuttle.get('transition_row_sum_max_error', float('nan')):.3g}"],
            ["Value-iteration Bellman residual", f"{shuttle.get('bellman_residual', float('nan')):.3g}"],
            ["Maximum VI vs PI value difference", f"{shuttle.get('max_vi_pi_value_difference', float('nan')):.3g}"],
            ["VI vs PI policy agreement", f"{100*shuttle.get('vi_pi_policy_agreement', float('nan')):.1f}%"],
            ["Finite t=0 vs VI policy agreement", f"{100*shuttle.get('finite_vi_policy_agreement', float('nan')):.1f}%"],
            ["VI / PI iterations", f"{shuttle.get('value_iteration_iterations')} / {shuttle.get('policy_iteration_iterations')}"],
            ["Uniform / deterministic-arrival threshold", f"{shuttle.get('value_iteration_threshold')} / {shuttle.get('deterministic_arrival_threshold')}"],
        ], [3.5 * inch, 2.9 * inch]),
        Spacer(1, 8),
        Paragraph("These tests are algorithmic rather than format checks: a small Bellman residual directly verifies the fixed-point equation, and exact agreement between independently evaluated policies makes a shared implementation error less likely. Replacing random arrivals by deterministic A=3 keeps the threshold at 13, which is the expected qualitative stability when only arrival variance is removed.", styles["Body"]),
    ])
    story += _figure(artifacts / "shuttle/finite_value.png", "Figure 1. Optimal finite-horizon value at t=0 for T=500.", styles)
    story += _figure(artifacts / "shuttle/infinite_value.png", "Figure 2. Optimal infinite-horizon value from value iteration.", styles)
    story += _figure(artifacts / "shuttle/optimal_policy.png", "Figure 3. Policy-iteration solution; dispatch begins at 13 waiting passengers.", styles)
    story.extend([
        Paragraph("Edge cases", styles["H2x"]),
        Paragraph("- At s=200, transition probability remains normalized and all overflow is represented by state 200.<br/>- For s&lt;15, dispatch empties the available queue; it never creates a negative queue.<br/>- T=500 uses a documented zero terminal cost; comparisons use t=0, where terminal effects have decayed.<br/>- Invalid discount factors, empty arrival supports, nonpositive capacity, and malformed policies are rejected.<br/>- Ties are resolved consistently by NumPy argmin, choosing wait (action 0).", styles["BulletX"]),
        PageBreak(),
        Paragraph("2. Policy gradient method", styles["H1x"]),
        Paragraph("The policy network outputs logits for a categorical action distribution. A trajectory is sampled until terminated or truncated. For time t, G_t=sum_(k=t) gamma^(k-t) r_k. Basic REINFORCE minimizes -sum log pi(a_t|s_t) times normalized G_t. Adam uses learning rate 0.01 for CartPole as specified. For Pong I use 0.001 because the high-dimensional input made 0.01 unstable; gamma is 0.99 and the policy is restricted to RIGHT/LEFT (ALE actions 2/3).", styles["Body"]),
        Paragraph("Pong frames are cropped to rows 35:195, downsampled by two, reduced to the first channel, backgrounds 144 and 109 removed, and all remaining pixels binarized. Consecutive preprocessed frames are differenced so direction of motion is observable. The policy is a 6400-200-2 MLP. CartPole uses a 4-128-2 MLP.", styles["Body"]),
        Paragraph("Baseline variant", styles["H2x"]),
        Paragraph("I chose a nonconstant learned state-value baseline V_phi(s), trained by mean squared error against Monte Carlo return. The policy advantage is G_t-V_phi(s_t), detached while updating the policy. This retains the unbiased score-function form while reducing variance when the value estimate is informative. Policy and value gradients are clipped to norm 5.", styles["Body"]),
        Paragraph("Policy-gradient edge cases and safeguards", styles["H2x"]),
        Paragraph("- Both terminated and truncated flags end trajectories.<br/>- Constant or single-step returns normalize to zeros instead of NaN.<br/>- Each episode receives a distinct deterministic reset seed; evaluation uses disjoint seeds.<br/>- Pong input shape is validated before slicing, and only action indices 0/1 can map to actions 2/3.<br/>- Frame history resets at every episode, preventing cross-episode leakage.<br/>- Greedy evaluation removes sampling variance from the deployed policy while retaining environment variance.<br/>- Raw rewards and configuration are saved, so reported means and standard deviations can be recomputed.", styles["BulletX"]),
        PageBreak(),
        Paragraph("3. CartPole-v1 results", styles["H1x"]),
        _metrics_table([
            ["Variant", "Train eps.", "Eval eps.", "Mean", "Std. dev.", "Last-100 train mean"],
            ["Basic", cart_basic.get("training_episodes"), cart_basic.get("evaluation_episodes"), f"{cart_basic.get('evaluation_mean', float('nan')):.3f}", f"{cart_basic.get('evaluation_std', float('nan')):.3f}", f"{cart_basic.get('last_100_training_mean', float('nan')):.2f}"],
            ["Learned baseline", cart_base.get("training_episodes"), cart_base.get("evaluation_episodes"), f"{cart_base.get('evaluation_mean', float('nan')):.3f}", f"{cart_base.get('evaluation_std', float('nan')):.3f}", f"{cart_base.get('last_100_training_mean', float('nan')):.2f}"],
        ], [1.2*inch, .75*inch, .75*inch, .8*inch, .8*inch, 1.45*inch]),
        Spacer(1, 7),
        Paragraph("Basic REINFORCE reaches near-ceiling greedy performance over 500 rollouts. The learned baseline reaches the 500-step environment limit on every recorded evaluation, giving mean 500 and zero standard deviation. The last-100 training mean is also higher with the baseline (426.37 vs 352.58), quantitative evidence consistent with improved learning stability. Because seeds differ, this is evidence rather than a definitive multi-seed causal estimate; the three-month plan addresses that limitation.", styles["Body"]),
    ])
    story += _figure(artifacts / "cartpole/basic/training_curve.png", "Figure 4. CartPole basic REINFORCE training rewards and 100-episode moving average.", styles)
    story += _figure(artifacts / "cartpole/basic/evaluation_histogram.png", "Figure 5. Basic REINFORCE reward distribution over 500 evaluation episodes.", styles)
    story += _figure(artifacts / "cartpole/baseline/training_curve.png", "Figure 6. CartPole REINFORCE with learned value baseline.", styles)
    story += _figure(artifacts / "cartpole/baseline/evaluation_histogram.png", "Figure 7. Baseline-agent reward distribution over 500 evaluation episodes.", styles)
    story.extend([
        PageBreak(),
        Paragraph("4. Pong-v5 results and execution status", styles["H1x"]),
        Paragraph("The implementation uses the requested Pong-v5 dynamics through Gymnasium's canonical ALE/Pong-v5 registration and restricts the learned policy to RIGHT and LEFT. The following artifacts validate environment creation, preprocessing, trajectory collection, both optimizer paths, checkpointing, evaluation, and plotting. They are intentionally labeled smoke runs because five truncated episodes cannot establish convergence.", styles["Note"]),
        _metrics_table([
            ["Variant", "Train eps.", "Eval eps.", "Step cap", "Mean", "Std. dev.", "Status"],
            ["Basic", pong_basic.get("training_episodes"), pong_basic.get("evaluation_episodes"), pong_basic.get("max_steps"), f"{pong_basic.get('evaluation_mean', float('nan')):.2f}", f"{pong_basic.get('evaluation_std', float('nan')):.2f}", pong_basic.get("status")],
            ["Learned baseline", pong_base.get("training_episodes"), pong_base.get("evaluation_episodes"), pong_base.get("max_steps"), f"{pong_base.get('evaluation_mean', float('nan')):.2f}", f"{pong_base.get('evaluation_std', float('nan')):.2f}", pong_base.get("status")],
        ], [1.2*inch, .7*inch, .7*inch, .7*inch, .75*inch, .75*inch, .75*inch]),
        Spacer(1, 7),
        Paragraph("The negative rewards are expected from an essentially untrained Pong agent and should not be used to compare baseline quality. For the required final experiment, run the two documented 10,000-episode commands without --max-steps. Each defaults to 500 full evaluation episodes and overwrites this section's artifact inputs; then regenerate the report.", styles["Body"]),
    ])
    story += _figure(artifacts / "pong/basic/training_curve.png", "Figure 8. Pong basic REINFORCE smoke-run rewards.", styles)
    story += _figure(artifacts / "pong/basic/evaluation_histogram.png", "Figure 9. Pong basic smoke evaluation; 20 truncated episodes.", styles)
    story += _figure(artifacts / "pong/baseline/training_curve.png", "Figure 10. Pong baseline smoke-run rewards.", styles)
    story += _figure(artifacts / "pong/baseline/evaluation_histogram.png", "Figure 11. Pong baseline smoke evaluation; 20 truncated episodes.", styles)
    story.extend([
        PageBreak(),
        Paragraph("5. Validation experiment instructions", styles["H1x"]),
        Paragraph("Run PYTHONPATH=src pytest -q first. This executes hand-calculated return checks, stochastic-matrix/boundary checks, independent DP-solver agreement, deterministic-arrival sensitivity, preprocessing invariants, and action restriction. Then run the shuttle command and inspect artifacts/shuttle/summary.json: the Bellman residual should be near numerical precision and VI/PI policy agreement should be 1. Run each policy-gradient command and use training_rewards.csv to verify that the 100-episode moving average rises; use evaluation_rewards.csv to recompute mean and population standard deviation. A random or untrained-policy control can be evaluated with the same seeds to distinguish learning from unusually easy episodes.", styles["Body"]),
        Paragraph("For targeted trajectory inspection, set --episodes 1 and add a temporary trace around the first ten environment steps recording state summary, logits, action, reward, return, and advantage. For Pong, additionally save the raw frame, processed frame, and frame difference. Expected behavior is finite logits, action IDs only 2/3, binary processed pixels, and returns that propagate point rewards backward by gamma=0.99.", styles["Body"]),
        Paragraph("6. What I would do with three additional months", styles["H1x"]),
        Paragraph("I would run at least 20 independent seeds per algorithm/environment and report bootstrap confidence intervals for learning-curve area, final-100 reward, evaluation mean, and evaluation variance. For the baseline, I would compare constant, episode-mean, exponential-moving-average, learned Monte Carlo value, and generalized advantage estimation baselines while holding architecture and environment interactions fixed. I would measure gradient variance directly from repeated trajectory batches and relate it to sample efficiency.", styles["Body"]),
        Paragraph("For shuttle dispatch I would sweep capacity, cost ratio, discount, arrival distributions (including bursty and time-varying processes), queue cap, and event-order assumptions; verify monotonicity/threshold structure; compare against simulation estimates with confidence intervals; and stress the cap under overload. For CartPole I would test observation noise, action delays, altered gravity/pole length, and robustness on held-out seeds. For Pong I would complete GPU training, compare raw frames with frame differences and small convolutional policies, test sticky-action settings, evaluate no-op start distributions, and record human-normalized score and wall-clock/sample efficiency. I would preregister evaluation seeds, retain every config/checkpoint, and use paired statistical tests with correction for multiple comparisons.", styles["Body"]),
        Paragraph("7. Deliverable checklist", styles["H1x"]),
        Paragraph("- GitHub username, repository name, and clickable link: provided on title page.<br/>- Run instructions: provided here and in README.md.<br/>- Required shuttle plots: Figures 1-3.<br/>- Training curves plus 100-episode moving averages: Figures 4, 6, 8, 10.<br/>- Evaluation histograms, means, and standard deviations: Figures 5, 7, 9, 11 and result tables.<br/>- Edge cases, correctness evidence, and validation directions: Sections 1-5.<br/>- Baseline choice and implementation: Section 2.<br/>- Three-month extension experiments: Section 6.<br/>- All source code and raw evidence: linked repository.", styles["BulletX"]),
    ])
    doc.build(story, onFirstPage=_page, onLaterPages=_page)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    parser.add_argument("--output", type=Path, default=Path("output/pdf/assignment1_report.pdf"))
    args = parser.parse_args(argv)
    build_report(args.artifacts, args.output)
    print(args.output)


if __name__ == "__main__":
    main()
