"""Custom CSS and dark glassmorphic styling tokens for the SignalGuard UI."""

from __future__ import annotations

import streamlit as st


def inject_custom_css() -> None:
    """Inject custom styles for a modern, glassmorphic dark interface."""
    st.markdown(
        """
        <style>
        .reportview-container .main .block-container {
            max-width: 1200px;
            padding-top: 1.5rem;
            padding-bottom: 3rem;
        }
        .main-header {
            background: linear-gradient(135deg, rgba(30, 41, 59, 0.9) 0%, rgba(15, 23, 42, 0.9) 100%);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 12px;
            padding: 1.5rem 2rem;
            margin-bottom: 1.5rem;
            box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        }
        .main-header h1 {
            color: #38bdf8;
            font-size: 2.2rem;
            margin: 0;
            font-weight: 700;
            letter-spacing: -0.5px;
        }
        .main-header p {
            color: #94a3b8;
            font-size: 1.05rem;
            margin: 0.4rem 0 0 0;
        }
        .badge-pill {
            display: inline-block;
            padding: 0.25rem 0.6rem;
            border-radius: 9999px;
            font-size: 0.75rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-right: 0.4rem;
        }
        .badge-dsp { background: rgba(56, 189, 248, 0.2); color: #38bdf8; border: 1px solid rgba(56, 189, 248, 0.4); }
        .badge-zero-ml { background: rgba(168, 85, 247, 0.2); color: #c084fc; border: 1px solid rgba(168, 85, 247, 0.4); }
        .badge-16k { background: rgba(34, 197, 94, 0.2); color: #4ade80; border: 1px solid rgba(34, 197, 94, 0.4); }

        .decision-banner {
            border-radius: 10px;
            padding: 1.2rem 1.6rem;
            margin: 1.2rem 0;
            border-left: 6px solid;
        }
        .decision-pass {
            background: rgba(16, 185, 129, 0.12);
            border-color: #10b981;
            color: #6ee7b7;
        }
        .decision-restored {
            background: rgba(245, 158, 11, 0.12);
            border-color: #f59e0b;
            color: #fcd34d;
        }
        .decision-reject-synth {
            background: rgba(239, 68, 68, 0.15);
            border-color: #ef4444;
            color: #fca5a5;
        }
        .decision-reject-unrec {
            background: rgba(225, 29, 72, 0.15);
            border-color: #e11d48;
            color: #fda4af;
        }
        .gauge-container {
            background: rgba(15, 23, 42, 0.6);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 8px;
            padding: 1rem;
            margin: 0.8rem 0;
        }
        .gauge-bar-bg {
            background: #334155;
            border-radius: 9999px;
            height: 14px;
            overflow: hidden;
            margin: 0.5rem 0;
        }
        .gauge-bar-fill {
            height: 100%;
            border-radius: 9999px;
            transition: width 0.5s ease-in-out;
        }
        .card-stat {
            background: rgba(30, 41, 59, 0.5);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 8px;
            padding: 0.8rem 1rem;
            text-align: center;
        }
        .card-stat .val {
            font-size: 1.5rem;
            font-weight: 700;
            color: #f8fafc;
        }
        .card-stat .lbl {
            font-size: 0.8rem;
            color: #94a3b8;
            text-transform: uppercase;
        }

        .anomaly-card {
            background: linear-gradient(135deg, rgba(30, 41, 59, 0.75) 0%, rgba(15, 23, 42, 0.85) 100%);
            border: 1px solid rgba(239, 68, 68, 0.25);
            border-left: 4px solid #ef4444;
            border-radius: 8px;
            padding: 1rem 1.25rem;
            margin-bottom: 0.85rem;
            box-shadow: 0 4px 14px rgba(0, 0, 0, 0.2);
            transition: transform 0.15s ease, border-color 0.15s ease;
        }
        .anomaly-card:hover {
            border-color: rgba(239, 68, 68, 0.5);
            transform: translateY(-1px);
        }
        .anomaly-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 0.4rem;
        }
        .anomaly-badge {
            display: inline-block;
            background: rgba(239, 68, 68, 0.18);
            color: #fca5a5;
            border: 1px solid rgba(239, 68, 68, 0.35);
            font-size: 0.75rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            padding: 0.2rem 0.55rem;
            border-radius: 4px;
        }
        .anomaly-title {
            color: #f1f5f9;
            font-size: 0.96rem;
            font-weight: 600;
            margin: 0 0 0.3rem 0;
            line-height: 1.4;
        }
        .anomaly-context {
            color: #94a3b8;
            font-size: 0.82rem;
            margin: 0;
            line-height: 1.35;
        }

        .clean-evidence-card {
            background: linear-gradient(135deg, rgba(16, 185, 129, 0.10) 0%, rgba(6, 78, 59, 0.15) 100%);
            border: 1px solid rgba(16, 185, 129, 0.3);
            border-left: 5px solid #10b981;
            border-radius: 8px;
            padding: 1.2rem 1.5rem;
            margin: 0.8rem 0;
            color: #a7f3d0;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
