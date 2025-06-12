import streamlit as st
import pandas as pd
import os
import tempfile
import time
import traceback


st.set_page_config(
    page_title="CCDA Data Quality Analyzer",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Import backend
try:
    from dataquality_analyzer import CCDASectionScorer

    BACKEND_AVAILABLE = True
except ImportError:
    BACKEND_AVAILABLE = False


# ========================================================================
# UI PATIENT INFORMATION DISPLAY FIX
# Changes needed in your streamlit_app.py
# ========================================================================

# REPLACE your existing display_document_details function with this enhanced version:

def display_document_details(score_report):
    """Display document and patient details with enhanced patient information extraction"""
    st.header("📋 Document Information")

    # Create columns for better layout
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("📄 Document Details")
        doc_id = score_report.get('document_id', 'Unknown')
        timestamp = score_report.get('timestamp', 'Unknown')

        # Better timestamp formatting
        if timestamp != 'Unknown' and 'T' in timestamp:
            try:
                from datetime import datetime
                dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                formatted_date = dt.strftime("%B %d, %Y")
                formatted_time = dt.strftime("%I:%M %p")
                display_timestamp = f"{formatted_date} at {formatted_time}"
            except:
                display_timestamp = timestamp.split('T')[0] if 'T' in timestamp else timestamp
        else:
            display_timestamp = timestamp

        st.info(f"""
        **📄 Document ID:** {doc_id}\n
        **📅 Analysis Date:** {display_timestamp}
        """)

    with col2:
        st.subheader("👤 Patient Information")

        # Enhanced patient information extraction
        patient_id = score_report.get('patient_id', 'Unknown')
        demographics = score_report.get('patient_demographics', {})

        # Extract patient information with fallbacks
        patient_name = demographics.get('patient_name', 'Not Available')
        age = demographics.get('age')
        gender = demographics.get('gender', 'Unknown')
        birth_date = demographics.get('formatted_birth_date')

        # Format age display
        if age is not None:
            age_display = f"{age} years old"
            if birth_date:
                age_display += f"\n(Born: {birth_date})"
        elif birth_date:
            age_display = f"Born: {birth_date}"
        else:
            age_display = "Age not available"

        # Format gender with icon
        if gender == 'Male':
            gender_display = "♂️ Male"
        elif gender == 'Female':
            gender_display = "♀️ Female"
        elif gender == 'Unknown':
            gender_display = "❓ Unknown"
        else:
            gender_display = f"👤 {gender}"

        # Display patient info with better formatting
        if patient_name != 'Not Available':
            st.success(f"""
            **👤 Patient Name:** {patient_name}\n
            **🎂 Age:** {age_display}\n
            **👤 Gender:** {gender_display}
            """)
        else:
            st.warning(f"""
            **👤 Patient Name:** ⚠️ Not found in document
            **🎂 Age:** {age_display}
            **👤 Gender:** {gender_display}
            """)

        # Debug information (optional - remove in production)
        with st.expander("🔍 Debug Patient Data"):
            st.write("**Raw demographics data:**")
            st.json(demographics)

    with col3:
        st.subheader("🤖 Analysis Configuration")
        config = score_report.get('analysis_config', {})

        if config.get('ai_enabled'):
            st.success("✨ ** LLM Enhanced**")
            st.info("🧠 ALL clinical reasoning via AI")
            # st.info("📚 Knowledge Base: NONE")
        else:
            st.warning("📋 **Rule-Based Only**")
            st.info("🔧 Basic analysis mode")


def display_patient_sidebar(score_report):
    """Display patient information in sidebar for easy reference"""

    demographics = score_report.get('patient_demographics', {})

    # Only show if we have patient data
    if demographics and any(demographics.values()):
        st.sidebar.markdown("---")
        st.sidebar.subheader("👤 Patient Summary")

        # Patient name
        patient_name = demographics.get('patient_name')
        if patient_name:
            st.sidebar.markdown(f"**Name:** {patient_name}")
        else:
            st.sidebar.warning("Name: Not found")

        # Age
        age = demographics.get('age')
        birth_date = demographics.get('formatted_birth_date')

        if age is not None:
            st.sidebar.markdown(f"**Age:** {age} years")
            if birth_date:
                st.sidebar.markdown(f"**Born:** {birth_date}")
        elif birth_date:
            st.sidebar.markdown(f"**Born:** {birth_date}")
        else:
            st.sidebar.warning("Age: Not found")

        # Gender
        gender = demographics.get('gender')
        if gender and gender != 'Unknown':
            gender_icon = "♂️" if gender == "Male" else "♀️" if gender == "Female" else "👤"
            st.sidebar.markdown(f"**Gender:** {gender_icon} {gender}")
        else:
            st.sidebar.warning("Gender: Not found")

def display_pure_llm_system_overview(score_report):
    """Display Pure LLM system capabilities overview"""
    # st.header("🤖 Pure LLM Analysis System")

    config = score_report.get('analysis_config', {})

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🎯 Complete LLM Coverage")

        sections = score_report.get('sections', {})
        sections_analyzed = sum(1 for s in sections.values() if s.get('section_present', False))

        st.metric("Sections Analyzed", f"{sections_analyzed}/3")
        st.metric("Dimensions per Section", "4")
        st.metric("Total LLM Analyses", f"{sections_analyzed * 4}")

        # Show section coverage
        coverage_status = []
        section_map = {
            'lab_results': '🔬 Lab Results',
            'medications': '💊 Medications',
            'problems': '🏥 Problems'
        }

        for section_key, section_name in section_map.items():
            if sections.get(section_key, {}).get('section_present', False):
                coverage_status.append(f"✅ {section_name}")
            else:
                coverage_status.append(f"❌ {section_name}")

        for status in coverage_status:
            st.write(status)

    with col2:
        st.subheader("🧠 Zero Knowledge Base")

        st.success("**AI Reasoning:**")
        st.write("• 🔬 Lab analysis without reference ranges")
        st.write("• 💊 Drug interactions without databases")
        st.write("• 🏥 Diagnostic assessment without rules")
        st.write("• 📊 Quality evaluation via LLM expertise")

        st.info("**All 4 Quality Dimensions:**")
        st.write("• 📊 Completeness")
        st.write("• 🏗️ Structural Integrity")
        st.write("• 🧠 Clinical Plausibility")
        st.write("• 📝 Narrative Consistency")


def display_dimension_analysis(section_data, section_name):
    """Display detailed dimension analysis with Pure LLM insights"""
    if not section_data or not section_data.get('section_present', False):
        return

    st.subheader(f"🔍 {section_name} - All Dimensions Analysis")

    category_scores = section_data.get('category_scores', {})

    if not category_scores:
        st.warning("No dimension scores available")
        return

    # Create tabs for each dimension
    dimension_tabs = st.tabs([
        "📊 Completeness",
        "🏗️ Structure",
        "🧠 Clinical",
        "📝 Narrative"
    ])

    dimensions = [
        ('completeness', '📊', 'Completeness'),
        ('structural_integrity', '🏗️', 'Structural Integrity'),
        ('clinical_plausibility', '🧠', 'Clinical Plausibility'),
        ('narrative_consistency', '📝', 'Narrative Consistency')
    ]

    for i, (dimension_key, emoji, dimension_name) in enumerate(dimensions):
        with dimension_tabs[i]:
            score = category_scores.get(dimension_key, 0)

            # Score display
            col1, col2 = st.columns([1, 3])
            with col1:
                if score >= 85:
                    st.success(f"### {score:.1f}/100")
                    st.success("Excellent")
                elif score >= 70:
                    st.warning(f"### {score:.1f}/100")
                    st.warning("Good")
                elif score >= 50:
                    st.error(f"### {score:.1f}/100")
                    st.error("Poor")
                else:
                    st.error(f"### {score:.1f}/100")
                    st.error("Critical")

            with col2:
                st.progress(max(0.0, min(1.0, score / 100)))
                st.write(f"**{emoji} {dimension_name}**")
                # st.write("🤖 Pure LLM Analysis - No hardcoded rules")

            # Show LLM-specific insights for this dimension
            display_dimension_insights(section_data, dimension_key, section_name)


def display_dimension_insights(section_data, dimension, section_name):
    """Display Pure LLM insights for specific dimension"""
    issues = section_data.get('issues', [])
    recommendations = section_data.get('recommendations', [])

    # Filter issues and recommendations for this dimension
    dimension_keywords = get_dimension_keywords(dimension)

    dimension_issues = []
    dimension_recs = []

    # More sophisticated filtering
    for issue in issues:
        issue_upper = issue.upper()
        if (dimension.upper() in issue_upper or
                any(keyword in issue_upper for keyword in dimension_keywords) or
                has_dimension_context(issue, dimension)):
            dimension_issues.append(issue)

    for rec in recommendations:
        rec_upper = rec.upper()
        if (dimension.upper() in rec_upper or
                any(keyword in rec_upper for keyword in dimension_keywords) or
                has_dimension_context(rec, dimension)):
            dimension_recs.append(rec)

    # If no specific findings, show general ones
    if not dimension_issues and issues:
        dimension_issues = issues[:2]  # Show first 2 general issues
    if not dimension_recs and recommendations:
        dimension_recs = recommendations[:2]  # Show first 2 general recommendations

    if dimension_issues:
        st.subheader("🔍 LLM Findings")
        for issue in dimension_issues[:5]:  # Show top 5
            if "🔴" in issue or "CRITICAL" in issue.upper():
                st.error(f"🔴 {issue}")
            elif "⚠️" in issue or "ABNORMAL" in issue.upper():
                st.warning(f"⚠️ {issue}")
            elif "✅" in issue or "NORMAL" in issue.upper():
                st.success(f"✅ {issue}")
            else:
                st.info(f" {issue}")

    if dimension_recs:
        st.subheader("💡 LLM Recommendations")
        for rec in dimension_recs[:3]:  # Show top 3
            st.info(f" {rec}")

    if not dimension_issues and not dimension_recs:
        st.info(f" No specific {dimension.replace('_', ' ')} findings available")


def get_dimension_keywords(dimension):
    """Get keywords for filtering dimension-specific content"""
    keywords_map = {
        'completeness': ['COMPLETENESS', 'MISSING', 'INCOMPLETE', 'COMPREHENSIVE', 'LACKING'],
        'structural_integrity': ['STRUCTURE', 'FORMAT', 'ORGANIZATION', 'INTEGRITY', 'FORMATTING'],
        'clinical_plausibility': ['CLINICAL', 'PLAUSIBILITY', 'INTERACTION', 'ABNORMAL', 'CRITICAL', 'DRUG', 'LAB'],
        'narrative_consistency': ['NARRATIVE', 'CONSISTENCY', 'DISCREPANCY', 'TEXT', 'DESCRIPTION']
    }
    return keywords_map.get(dimension, [])


def has_dimension_context(text, dimension):
    """Check if text has contextual relevance to dimension"""
    text_upper = text.upper()

    context_patterns = {
        'completeness': ['NOT FOUND', 'ABSENT', 'UNAVAILABLE'],
        'structural_integrity': ['POORLY FORMATTED', 'MISSING UNITS', 'INVALID'],
        'clinical_plausibility': ['DANGEROUSLY', 'CONTRAINDICATED', 'RISK'],
        'narrative_consistency': ['MISMATCH', 'DIFFERS', 'INCONSISTENT']
    }

    patterns = context_patterns.get(dimension, [])
    return any(pattern in text_upper for pattern in patterns)


def deduplicate_recommendations(recommendations):
    """Remove duplicate recommendations while preserving order"""
    if not recommendations:
        return []

    seen = set()
    unique_recommendations = []

    for rec in recommendations:
        clean_rec = rec.strip()
        normalized = clean_rec.lower()

        # Remove common prefixes for better duplicate detection
        for prefix in ['🔍', '💡', '📋', '🔧', '🚨', '⚠️', '✅', '🤖']:
            normalized = normalized.replace(prefix, '').strip()

        # Remove "emergency:", "ai recommendation:", etc. for comparison
        comparison_text = normalized
        for prefix in ['emergency:', 'ai recommendation:', 'recommendation:', 'consider', 'monitor for']:
            if comparison_text.startswith(prefix):
                comparison_text = comparison_text[len(prefix):].strip()
                break

        if comparison_text not in seen and comparison_text:
            seen.add(comparison_text)
            unique_recommendations.append(clean_rec)

    return unique_recommendations


def deduplicate_findings(findings):
    """Remove duplicate findings while preserving order"""
    if not findings:
        return []

    seen = set()
    unique_findings = []

    for finding in findings:
        clean_finding = finding.strip()
        normalized = clean_finding.lower()

        # Remove emojis and status indicators
        for indicator in ['🔴', '⚠️', '✅', '🤖', '📋', 'critical high:', 'critical low:',
                          'abnormal high:', 'abnormal low:', 'normal:', ' llm:']:
            normalized = normalized.replace(indicator, '').strip()

        # Extract the core content for comparison
        if '=' in normalized:
            core_content = normalized.split('=')[0].strip()
        else:
            core_content = normalized

        # Remove common prefixes
        for prefix in ['ai pattern:', 'drug interaction:', 'diagnosis conflict:', 'llm clinical reasoning:']:
            if core_content.startswith(prefix):
                core_content = core_content[len(prefix):].strip()
                break

        if core_content not in seen and core_content:
            seen.add(core_content)
            unique_findings.append(clean_finding)

    return unique_findings


def display_clinical_findings(section_data, section_name):
    """Display clinical findings with duplicate removal and Pure LLM emphasis"""
    if not section_data:
        return

    clinical_details = section_data.get('clinical_details', {})
    clinical_score = section_data.get('category_scores', {}).get('clinical_plausibility', 0)

    if clinical_details and clinical_details.get('findings'):
        st.subheader("🔬 LLM Clinical Analysis")

        # Emphasize Pure LLM capabilities
        col1, col2 = st.columns(2)
        with col1:
            if clinical_score >= 90:
                st.success(f"**Clinical Score: {clinical_score:.1f}/100**")
            elif clinical_score >= 75:
                st.warning(f"**Clinical Score: {clinical_score:.1f}/100**")
            else:
                st.error(f"**Clinical Score: {clinical_score:.1f}/100**")

        with col2:
            st.info("🤖 ** AI Clinical Reasoning**")
            # st.info("📚 **No Knowledge Base Used**")

        # Show findings with deduplication
        findings = clinical_details.get('findings', [])
        unique_findings = deduplicate_findings(findings)

        if unique_findings:
            with st.expander("📋 LLM Clinical Findings", expanded=True):
                for finding in unique_findings:
                    if "🔴" in finding or "CRITICAL" in finding.upper():
                        st.error(f"🔴 {finding}")
                    elif "⚠️" in finding or "ABNORMAL" in finding.upper():
                        st.warning(f"⚠️ {finding}")
                    elif "✅" in finding or "NORMAL" in finding.upper():
                        st.success(f"✅ {finding}")
                    else:
                        st.info(f"🤖 {finding}")

        # Show recommendations with deduplication
        recommendations = clinical_details.get('recommendations', [])
        unique_recommendations = deduplicate_recommendations(recommendations)

        if unique_recommendations:
            with st.expander("💡LLM Clinical Recommendations"):
                for rec in unique_recommendations:
                    if "🚨" in rec or "EMERGENCY" in rec.upper():
                        st.error(f"🚨 {rec}")
                    elif "⚠️" in rec or "CRITICAL" in rec.upper():
                        st.warning(f"⚠️ {rec}")
                    else:
                        st.info(f"🤖 {rec}")


def display_pure_llm_sidebar_summary(score_report):
    """Display Pure LLM system summary in sidebar"""
    # st.sidebar.markdown("---")
    # st.sidebar.subheader("🤖 Pure LLM System")

    # System capabilities
    # with st.sidebar.expander("🧠 AI Capabilities", expanded=True):
    #     st.write("**✅ Complete LLM Coverage:**")
    #     st.write("• 🔬 Lab Results Analysis")
    #     st.write("• 💊 Drug Interaction Detection")
    #     st.write("• 🏥 Diagnostic Assessment")
    #     st.write("• 📊 All 4 Quality Dimensions")
    #
    #     st.write("**🎯 Zero Knowledge Base:**")
    #     st.write("• No hardcoded reference ranges")
    #     st.write("• No drug interaction databases")
    #     st.write("• No diagnostic rule engines")
    #     st.write("• Pure AI clinical reasoning")

    # Analysis statistics
    with st.sidebar.expander("📊 Analysis Stats"):
        sections = score_report.get('sections', {})
        total_dimensions = 0
        llm_analyzed = 0

        for section_data in sections.values():
            if section_data.get('section_present', False):
                category_scores = section_data.get('category_scores', {})
                total_dimensions += len(category_scores)
                llm_analyzed += len(category_scores)

        st.metric("Total Dimensions", total_dimensions)
        st.metric("LLM Analyzed", llm_analyzed)
        st.metric("Knowledge Base Used", "0")

        config = score_report.get('analysis_config', {})
        if config.get('ai_enabled'):
            st.success("🤖 LLM Mode Active")
        else:
            st.warning("📋 Rule-Based Mode")


# Check backend availability
if not BACKEND_AVAILABLE:
    st.error("❌ Backend not available")
    st.error("Make sure 'dataquality_analyzer.py' is in the same directory")
    st.stop()

# App Header with Pure LLM branding
st.title(" CCDA Data Quality Analyzer")
offline_mode = False
force_cpu = False

# File Upload
st.header("📁 Upload CCDA Document")
uploaded_file = st.file_uploader("Choose a CCDA XML document", type="xml",
                                 help="Upload a CCDA XML file for comprehensive quality analysis")

if uploaded_file is not None:
    st.info(f"📁 **File:** {uploaded_file.name} ({uploaded_file.size:,} bytes)")

    # Progress indicators
    progress_bar = st.progress(0)
    status_text = st.empty()

    try:
        # Create temporary file
        status_text.text("📁 Processing file...")
        progress_bar.progress(20)

        with tempfile.NamedTemporaryFile(delete=False, suffix='.xml') as temp_file:
            temp_file.write(uploaded_file.getbuffer())
            temp_file_path = temp_file.name

        # Setup configuration
        status_text.text("⚙️ Configuring LLM analysis...")
        progress_bar.progress(40)

        config = {}
        if force_cpu:
            config['device'] = 'cpu'
        if offline_mode:
            config['use_ai_analysis'] = False

        # Create scorer
        status_text.text(" Loading LLM models...")
        progress_bar.progress(60)

        scorer = CCDASectionScorer(config)

        # Process document
        status_text.text("📊 Analyzing document with LLM...")
        progress_bar.progress(80)

        start_time = time.time()
        score_report = scorer.score_ccda(temp_file_path)
        processing_time = time.time() - start_time

        progress_bar.progress(100)
        status_text.text("✅ Analysis complete!")

        time.sleep(1)
        progress_bar.empty()
        status_text.empty()

        # Check for errors
        if 'error' in score_report:
            st.error(f"❌ Error: {score_report['error']}")
            st.info("💡 Try enabling 'Rule-based scoring only' in the sidebar")
            st.stop()

        # Success message
        st.success("🎉 LLM Analysis completed successfully!")
        st.info(f"⏱️ Processing time: {processing_time:.1f} seconds")

        # Display Pure LLM System Overview
        display_pure_llm_system_overview(score_report)

        # Add separator
        st.markdown("---")

        # Display Document Details
        display_document_details(score_report)

        # Add separator
        st.markdown("---")

        # Display Results
        st.header("📊 Quality Report")

        # Overall Score with Pure LLM branding
        overall_score = score_report.get('overall_score', 0)
        quality_level = score_report.get('quality_level', 'Unknown')

        col1, col2 = st.columns([1, 3])
        with col1:
            if quality_level == "High":
                st.success(f"## {overall_score:.1f}/100")
                st.success(f"🟢 {quality_level} Quality")
            elif quality_level == "Medium":
                st.warning(f"## {overall_score:.1f}/100")
                st.warning(f"🟡 {quality_level} Quality")
            else:
                st.error(f"## {overall_score:.1f}/100")
                st.error(f"🔴 {quality_level} Quality")

        with col2:
            st.progress(max(0.0, min(1.0, overall_score / 100)))
            analysis_config = score_report.get('analysis_config', {})
            if analysis_config.get('ai_enabled'):
                st.success("✨ **LLM Enhanced Analysis**")
                # st.info("🧠 All clinical reasoning via AI")
                st.info(
                    f"📊 Analyzed {len([s for s in score_report.get('sections', {}).values() if s.get('section_present', False)])} sections across 4 dimensions each")
            else:
                st.info("📋 **Rule-Based Analysis**")
                st.info("🔧 Basic analysis mode")

        # Enhanced Section Analysis
        st.header("🔍 Complete Section Analysis")
        st.subheader(" LLM Analysis for ALL Sections and ALL Dimensions")

        sections = score_report.get('sections', {})

        # Create enhanced tabs with better organization
        tab_labels = []
        tab_data = []

        section_info = {
            'lab_results': ('🔬', 'Lab Results'),
            'medications': ('💊', 'Medications'),
            'problems': ('🏥', 'Problems')
        }

        for section_key, (emoji, name) in section_info.items():
            section_data = sections.get(section_key, {})
            if section_data.get('section_present', False):
                score = section_data.get('overall_score', 0)
                tab_labels.append(f"{emoji} {name} ({score:.0f})")
                tab_data.append((section_key, section_data, name))
            else:
                tab_labels.append(f"{emoji} {name} (Missing)")
                tab_data.append((section_key, section_data, name))

        if tab_data:
            tabs = st.tabs(tab_labels)

            for i, (section_key, section_data, section_name) in enumerate(tab_data):
                with tabs[i]:
                    if section_data.get('section_present', False):
                        # Section overview
                        section_score = section_data.get('overall_score', 0)
                        col1, col2 = st.columns([1, 3])

                        with col1:
                            if section_score >= 80:
                                st.success(f"## {section_score:.1f}/100")
                            elif section_score >= 60:
                                st.warning(f"## {section_score:.1f}/100")
                            else:
                                st.error(f"## {section_score:.1f}/100")

                        with col2:
                            st.progress(max(0.0, min(1.0, section_score / 100)))
                            quality_level = section_data.get('quality_level', 'Unknown')
                            st.write(f"**Quality Level: {quality_level}**")
                            st.write("** LLM Analysis - No hardcoded rules**")

                        # Quick dimension overview
                        st.subheader("📊 Dimension Scores Overview")
                        category_scores = section_data.get('category_scores', {})
                        col1, col2, col3, col4 = st.columns(4)

                        dimension_info = [
                            ('completeness', 'Completeness', '📊'),
                            ('structural_integrity', 'Structure', '🏗️'),
                            ('clinical_plausibility', 'Clinical', '🧠'),
                            ('narrative_consistency', 'Narrative', '📝')
                        ]

                        for j, (dim_key, dim_name, emoji) in enumerate(dimension_info):
                            score = category_scores.get(dim_key, 0)
                            with [col1, col2, col3, col4][j]:
                                st.metric(f"{emoji} {dim_name}", f"{score:.1f}")

                        # Enhanced dimension analysis
                        display_dimension_analysis(section_data, section_name)

                        # Clinical findings (existing function enhanced)
                        display_clinical_findings(section_data, section_name)

                        # Issues summary
                        issues = section_data.get("issues", [])
                        if issues:
                            st.subheader("⚠️ Additional Issues")
                            with st.expander(f"View all {len(issues)} issues"):
                                for issue in issues:
                                    if "🔴" in issue or "CRITICAL" in issue.upper():
                                        st.error(f"🔴 {issue}")
                                    elif "⚠️" in issue or "ABNORMAL" in issue.upper():
                                        st.warning(f"⚠️ {issue}")
                                    else:
                                        st.info(f"ℹ️ {issue}")
                    else:
                        st.warning(f"📋 {section_name} section not present in document")
                        st.info("This section was not found in the CCDA document or contains no data.")

                        # Suggest what could be added
                        suggestions = {
                            'lab_results': "Consider adding laboratory test results and values",
                            'medications': "Consider adding current medications and prescriptions",
                            'problems': "Consider adding diagnoses and medical problems"
                        }

                        if section_key in suggestions:
                            st.info(f"💡 **Suggestion:** {suggestions[section_key]}")

        # Add Pure LLM sidebar summary
        display_pure_llm_sidebar_summary(score_report)

        # Export Options
        st.header("📤 Export Results")
        st.subheader("Download comprehensive analysis reports")

        col1, col2, col3 = st.columns(3)

        with col1:
            json_report = scorer.generate_json_report(score_report)
            st.download_button(
                label="📄 Download JSON Report",
                data=json_report,
                file_name=f"ccda_pure_llm_report_{uploaded_file.name}.json",
                mime="application/json",
                help="Complete analysis data in JSON format"
            )

        with col2:
            text_report = scorer.generate_report(score_report)
            st.download_button(
                label="📝 Download Text Report",
                data=text_report,
                file_name=f"ccda_pure_llm_report_{uploaded_file.name}.txt",
                mime="text/plain",
                help="Human-readable analysis report"
            )

        with col3:
            # Create summary CSV for quick analysis
            summary_data = {
                'Document_ID': [score_report.get('document_id', 'Unknown')],
                'Overall_Score': [overall_score],
                'Quality_Level': [quality_level],
                'Processing_Time_Seconds': [processing_time],
                'AI_Enabled': [analysis_config.get('ai_enabled', False)]
            }

            # Add section scores
            for section_key, section_data in sections.items():
                if section_data.get('section_present', False):
                    summary_data[f'{section_key.title()}_Score'] = [section_data.get('overall_score', 0)]
                    # Add dimension scores
                    category_scores = section_data.get('category_scores', {})
                    for dim_key, score in category_scores.items():
                        summary_data[f'{section_key.title()}_{dim_key.title()}_Score'] = [score]

            summary_df = pd.DataFrame(summary_data)
            csv_data = summary_df.to_csv(index=False)

            st.download_button(
                label="📊 Download CSV Summary",
                data=csv_data,
                file_name=f"ccda_summary_{uploaded_file.name}.csv",
                mime="text/csv",
                help="Analysis summary in CSV format for data analysis"
            )

    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        with st.expander("🔍 Error Details"):
            st.code(traceback.format_exc())

        # Suggest solutions
        st.subheader("💡 Troubleshooting")
        st.info("**Common solutions:**")
        st.write("• Try enabling 'Rule-based scoring only' in the sidebar")
        st.write("• Check if the CCDA file is valid XML")
        st.write("• Ensure you have the required API keys set for LLM analysis")
        st.write("• Verify the dataquality_analyzer.py file is in the same directory")

    finally:
        # Cleanup
        try:
            if 'temp_file_path' in locals() and os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
        except:
            pass

else:
    # Enhanced instructions when no file is uploaded
    st.info("👆 **Please upload a CCDA XML document to begin LLM analysis**")

    # API Configuration help
    if not offline_mode:
        with st.expander("🔑 API Configuration Help"):
            st.write("""
            **To enable LLM analysis, set environment variables:**

            ```bash
            # For OpenAI (recommended)
            export OPENAI_API_KEY="your-api-key-here"

            # For Anthropic Claude (future support)
            export ANTHROPIC_API_KEY="your-api-key-here"
            ```

            **Or create a .env file in the same directory:**
            ```
            OPENAI_API_KEY=your-api-key-here
            ANTHROPIC_API_KEY=your-api-key-here
            ```

            **Without API keys:** The system will fall back to rule-based analysis with limited capabilities.
            """)

