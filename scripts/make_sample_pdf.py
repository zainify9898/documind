"""Generate data/sample.pdf — a short research-style document about
artificial intelligence in healthcare, used for demos and tests.

Run:  python scripts/make_sample_pdf.py
"""

import os

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

TITLE = "Artificial Intelligence in Healthcare: Opportunities and Challenges"
AUTHOR = "DocuMind Sample Document — 2026"

SECTIONS = [
    (
        "1. Introduction",
        "Artificial intelligence (AI) is rapidly changing modern healthcare. "
        "Machine learning models now assist doctors with diagnosis, help "
        "hospitals manage patient flow, and accelerate the discovery of new "
        "medicines. This document summarises where AI is already useful in "
        "healthcare, what results have been measured, and which risks must be "
        "managed before these systems can be trusted at scale. The examples "
        "below come from peer-reviewed studies and published clinical trials, "
        "not from marketing claims, so that the numbers can be checked "
        "against the original sources.",
    ),
    (
        "2. AI in Medical Diagnosis",
        "The most mature application of AI in healthcare is medical imaging. "
        "Deep learning models trained on large sets of labelled scans can "
        "detect disease patterns that are difficult for the human eye to spot. "
        "In a landmark 2023 study, an AI system for detecting diabetic "
        "retinopathy from retinal photographs achieved 97.5 percent sensitivity "
        "and 93.4 percent specificity, matching the performance of specialist "
        "ophthalmologists. Similar systems are used for chest X-rays, where "
        "AI triage tools have reduced the average reporting time for urgent "
        "cases by 30 percent. In breast cancer screening, a 2020 study "
        "published in Nature showed that an AI system reduced false positives "
        "by 5.7 percent and false negatives by 9.4 percent compared with "
        "human radiologists. These tools do not replace doctors; they "
        "prioritise the most urgent cases so that clinicians can act faster, "
        "and every AI finding is still reviewed by a qualified professional.",
    ),
    (
        "3. AI in Drug Discovery",
        "Discovering a new medicine traditionally takes ten to fifteen years "
        "and costs more than two billion dollars. AI shortens this process in "
        "two ways. First, machine learning models predict how candidate "
        "molecules will behave, allowing researchers to screen millions of "
        "compounds in software before testing a few hundred in the laboratory. "
        "Second, generative models can propose entirely new molecular "
        "structures for a chosen target. In 2024, the first AI-designed drug "
        "candidate for idiopathic pulmonary fibrosis entered phase 2 clinical "
        "trials, reaching that stage in under 30 months — roughly half the "
        "usual time for early-stage discovery. Protein structure prediction "
        "has advanced just as fast: DeepMind's AlphaFold system has predicted "
        "the structures of more than 200 million proteins, a database that is "
        "now freely available to researchers worldwide and is used to design "
        "new treatments for diseases that were previously hard to target.",
    ),
    (
        "4. AI in Hospital Administration",
        "Beyond diagnosis and research, AI reduces the administrative burden "
        "on hospitals. Natural language processing tools transcribe doctor "
        "and patient conversations and draft clinical notes automatically; "
        "early deployments have cut documentation time by around 20 percent. "
        "Scheduling systems use machine learning to predict patient no-shows "
        "and to fill operating theatre slots more efficiently, while "
        "triage chatbots direct patients to the right department before they "
        "arrive. These applications matter because clinicians in many health "
        "systems report spending almost half of their working day on "
        "paperwork rather than on patients.",
    ),
    (
        "5. Risks and Limitations",
        "AI in healthcare also carries serious risks. Models trained on data "
        "from one hospital or one population often perform worse on other "
        "groups, a problem known as dataset bias, which can lead to unequal "
        "care. Patient data is highly sensitive, so privacy regulations such "
        "as the GDPR and HIPAA limit how data can be shared and stored. Many "
        "deep learning models are also difficult to explain: a doctor cannot "
        "always see why a model made a particular prediction, which makes "
        "clinical accountability difficult. Finally, regulatory approval is "
        "slow, and most health systems require extensive validation before an "
        "AI tool can be used on real patients. A model that works well in a "
        "research paper may fail in a busy clinic if the data, the equipment, "
        "or the patient population is different.",
    ),
    (
        "6. Conclusion",
        "AI already helps doctors diagnose disease earlier, shortens drug "
        "discovery, and reduces administrative work in hospitals. The "
        "measured gains — 97.5 percent sensitivity in diabetic retinopathy "
        "detection, 30 percent faster reporting of urgent X-rays, 5.7 percent "
        "fewer false positives in breast screening, and drug candidates "
        "reaching clinical trials in under 30 months — show that the "
        "technology is real, not hype. The remaining work is not only "
        "technical: fairer datasets, clearer explanations, stronger privacy, "
        "and careful regulation will decide whether AI becomes a reliable "
        "part of everyday healthcare.",
    ),
]


def main() -> None:
    out_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "sample.pdf")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(
        out_path,
        pagesize=A4,
        leftMargin=2 * cm,
        rightMargin=2 * cm,
        topMargin=2 * cm,
        bottomMargin=2 * cm,
        title=TITLE,
        author=AUTHOR,
    )
    story = [Paragraph(TITLE, styles["Title"]), Spacer(1, 0.4 * cm)]
    story.append(Paragraph(AUTHOR, styles["Normal"]))
    story.append(Spacer(1, 0.6 * cm))
    for heading, body in SECTIONS:
        story.append(Paragraph(heading, styles["Heading2"]))
        story.append(Paragraph(body, styles["BodyText"]))
        story.append(Spacer(1, 0.3 * cm))
    doc.build(story)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
