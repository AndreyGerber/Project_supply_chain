from collections import Counter

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from _utils import RATING_LABELS, STATIC_DIR, load_processed, remove_redundant, reply_mask

st.set_page_config(layout="wide")
st.title("🧹 Preprocessing")

# Columns that only exist for technical reasons and are not shown
HIDDEN_COLS = ["review_text_clean_light", "review_text_clean_advanced", "review_text_clean", "issue_categories",
               "rating_numeric", "rating_svg", "review_length", "domain", "language", "sentiment",
               "has_negation", "company_site"]
TIME_COLS = ["year", "month_name", "weekday", "season", "day_period"]
CUSTOM_STOPWORDS = {
    "order", "ordered", "get", "got", "received", "company", "part", "parts",
    "product", "service", "still", "even", "one", "would", "customer",
    "back", "said", "told", "review", "buy", "item", "items", "2",
}
STATUS_CSS = """<style>
.status-table { width: 100%; border-collapse: collapse; font-family: sans-serif; }
.status-table th, .status-table td { border-bottom: 1px solid #e6e9ef; padding: 12px; text-align: left; font-size: 16px; }
.status-table th { background-color: #f0f2f6; color: #31333F; font-weight: bold; }
.strikethrough { text-decoration: line-through; color: #9e9e9e; opacity: 0.7; font-style: italic; }
</style>"""


def status_table(columns, data, cleaned=(), dropped=()):
    """Shows which columns are kept, cleaned or dropped, with their number of unique values."""
    rows = ""
    for col in list(columns) + [c for c in dropped if c not in columns]:
        is_dropped = col in dropped
        unique = data[col].nunique() if col in data.columns and not is_dropped else "-"
        icon = "🗑️" if is_dropped else "🎯" if col == "rating" else "✅" if col in cleaned else "❌"
        css = ' class="strikethrough"' if is_dropped else ""
        rows += f"<tr{css}><td>{col}</td><td>{unique}</td><td>{icon}</td></tr>"
    st.markdown(f"{STATUS_CSS}<table class='status-table'><thead><tr><th>Column Name</th><th>Unique Values</th>"
                f"<th>Status</th></tr></thead><tbody>{rows}</tbody></table>", unsafe_allow_html=True)
    st.write("")


def heatmap(data, x_label, y_label, color_label, scale, text_auto, title, **layout):
    fig = px.imshow(data, labels=dict(x=x_label, y=y_label, color=color_label), color_continuous_scale=scale,
                    text_auto=text_auto, aspect="auto", title=title)
    fig.update_layout(**layout)
    return fig


def row_percent(table):
    return table.div(table.sum(axis=1), axis=0) * 100


def column_percent(table):
    return table.div(table.sum(axis=0), axis=1) * 100


# =========================
# Load data
# =========================
df = load_processed().drop(columns=["domain", "language", "sentiment", "has_negation", "company_site", "rating_svg"],
                           errors="ignore")
display_cols = [c for c in df.columns if c not in HIDDEN_COLS]

st.success(f"✅ Successfully linked to the dataset! ({len(df)} rows loaded)")
with st.expander("🔍 View Raw Data Columns"):
    st.write("Current relevant columns in our dataset:")
    st.code(display_cols)
    st.subheader("Data Preview (First 10 rows)")
    st.dataframe(df[display_cols].head(10), width="stretch")

st.write("### 📋 Dataset Column Overview")
overview = pd.DataFrame({"Column Name": [c for c in ["rating", "review_text", "date", "location", "verified",
                                                    "company"] if c in df.columns]})
overview["Unique Values (nunique)"] = overview["Column Name"].map(lambda c: df[c].nunique())
st.dataframe(overview, hide_index=True, width="stretch")

# =========================
# Duplicates
# =========================
st.subheader("🔍 Deep Dive: Why are there duplicates in 'review_text'?")
st.code(f"""Successfully linked to the dataset! ({len(df)} rows loaded).
But only {df['review_text'].nunique()} unique entries in "review_text".
Found {len(df) - df['review_text'].nunique()} duplicates.""", language="python")

_, img_col, _ = st.columns([1, 2, 1])
img_col.image(str(STATIC_DIR / "what_is_it.png"), width="stretch")

is_reply = reply_mask(df)
system_replies = df[is_reply]
df_no_system = df[~is_reply]

st.write(f"**A. System Replies:** Found {len(system_replies)} rows that are just company responses.")
st.code("""system_replies = df[df['review_text'].str.contains(r"^Reply from", na=False, case=False, regex=True)]""",
        language="python")

if not system_replies.empty:
    st.markdown("#### 🏢 Summary of System Replies by Company")
    company_summary = (system_replies.groupby("company")["review_text"]
                       .agg(Count="count", Example="first").reset_index()
                       .sort_values("Count", ascending=False))
    st.dataframe(company_summary, hide_index=True, width="stretch")
    st.info(f"💡 **Insight:** Instead of showing all {len(system_replies)} rows, we summarized them by company.")

unique_user_count = df_no_system["review_text"].nunique()
extra_rows = len(df_no_system) - unique_user_count
total_identified = len(system_replies) + extra_rows

st.write(f"**B. Genuine Comment Duplicates:** Identified {extra_rows} extra copies of customer phrases.")
text_counts = df_no_system["review_text"].value_counts()
real_duplicates = text_counts[text_counts > 1].rename_axis("Review Content").reset_index(name="Occurrence Count")
if not real_duplicates.empty:
    st.dataframe(real_duplicates.head(10), width="stretch", hide_index=True)

st.info(f"""
💡 **Conclusion:** We have identified all **{total_identified}** redundant entries:
* **{len(system_replies)}** are automated system replies (starting with 'Reply from').
* **{extra_rows}** are extra copies of common customer phrases.

Total: {len(system_replies)} + {extra_rows} = **{total_identified}**.
This explains why we have {len(df)} total rows but only **{unique_user_count}** unique customer comments.
""")

# =========================
# Cleaning
# =========================
st.subheader("🧹 Data Cleaning: Removing Redundant Data")
df = remove_redundant(df).copy()
st.warning(f"""
**Cleaning Summary:**
Removed a total of **{total_identified}** redundant rows.
- **{len(system_replies)}** automated company replies were deleted.
- **{extra_rows}** duplicate customer comments were removed.

Current dataset size: **{len(df)}** unique reviews.
""")

st.write("### 📋 Preprocessing Status Overview")
status_table(display_cols, df, cleaned=["review_text"])
st.divider()

# =========================
# Date features
# =========================
st.header("🚀 Lets work on our date-data")

SEASONS = {12: "Winter", 1: "Winter", 2: "Winter", 3: "Spring", 4: "Spring", 5: "Spring",
           6: "Summer", 7: "Summer", 8: "Summer", 9: "Autumn", 10: "Autumn", 11: "Autumn"}
hour = df["date"].dt.hour
df_time = df.assign(
    year=df["date"].dt.year,
    month_name=df["date"].dt.month_name(),
    weekday=df["date"].dt.day_name(),
    season=df["date"].dt.month.map(SEASONS).fillna("Unknown"),
    day_period=pd.cut(hour, bins=[-1, 4, 11, 16, 20, 23],
                      labels=["Night", "Morning", "Afternoon", "Evening", "Night"], ordered=False)
    .astype(str).where(hour.notna(), "Unknown"),
).drop(columns=["date"])
df_time = df_time[TIME_COLS + [c for c in df_time.columns if c not in TIME_COLS]]

st.write("### 🚀 Results (Cleaned Preview)")
st.dataframe(df_time.drop(columns=["review_length", "review_text_clean_light", "review_text_clean_advanced"],
                          errors="ignore").head(15), width="stretch")

st.divider()
st.header("🕵️ Correlation Analysis")
col1, col2 = st.columns(2)

with col1:
    st.subheader("1. Rating by Day Period")
    rating_period = (df_time.groupby("day_period")["rating"].mean()
                     .reindex(["Morning", "Afternoon", "Evening", "Night"]).dropna().reset_index())
    fig_day = px.bar(rating_period, x="day_period", y="rating", text=rating_period["rating"].round(2),
                     color="rating", color_continuous_scale="RdYlGn", range_color=[1, 5],
                     title="Average Rating per Time of Day")
    fig_day.update_traces(textposition="outside")
    fig_day.update_layout(yaxis_range=[1, 5.5], height=500, margin=dict(l=50, r=50, t=80, b=50))
    st.plotly_chart(fig_day, width="stretch")

with col2:
    st.subheader("2. Rating Trend & Review Volume by Year")
    year_stats = df_time.groupby("year")["rating"].agg(avg_rating="mean", review_count="count").reset_index()
    fig_combined = go.Figure([
        go.Bar(x=year_stats["year"], y=year_stats["review_count"], name="Number of Reviews",
               marker_color="rgba(100, 149, 237, 0.3)", yaxis="y2"),
        go.Scatter(x=year_stats["year"], y=year_stats["avg_rating"], name="Avg Rating",
                   mode="lines+markers+text", line=dict(color="firebrick", width=3),
                   text=year_stats["avg_rating"].round(2), textposition="top center"),
    ])
    fig_combined.update_layout(
        title="Avg Rating vs. Volume per Year", height=500, margin=dict(l=50, r=50, t=80, b=50),
        xaxis=dict(type="category", title="Year"),
        yaxis=dict(title="Average Rating", range=[1, 5]),
        yaxis2=dict(title="Number of Reviews", overlaying="y", side="right", range=[0, 2500], dtick=250,
                    showgrid=False),
        legend=dict(x=0.3, y=0.3, bgcolor="rgba(255,255,255,0.6)"),
    )
    st.plotly_chart(fig_combined, width="stretch")

for title, column, order, offset in [
    ("### 🌡️ Seasonal Rating Distribution", "season", ["Spring", "Summer", "Autumn", "Winter"], 3),
    ("### 📅 Monthly Rating Distribution", "month_name",
     ["January", "February", "March", "April", "May", "June",
      "July", "August", "September", "October", "November", "December"], 5),
]:
    st.markdown(title)
    name = "Season" if column == "season" else "Month"
    counts = (pd.crosstab(df_time["rating"], df_time[column])
              .reindex(columns=order, fill_value=0).sort_index(ascending=False))
    extra = {} if column == "season" else dict(xaxis_tickangle=-45, height=500)
    left, right = st.columns(2)
    left.subheader(f"{offset}. {'Absolute Volume' if column == 'season' else 'Monthly Volume'} (Counts)")
    left.plotly_chart(heatmap(counts, name, "Rating", "Count", "YlGnBu", True,
                              f"Total Reviews (Rating vs. {name})", **extra), width="stretch")
    right.subheader(f"{offset + 1}. {'Relative' if column == 'season' else 'Monthly Relative'} Distribution (%)")
    right.plotly_chart(heatmap(column_percent(counts), name, "Rating", "Percentage (%)", "Viridis", ".1f",
                               f"Percentage of Ratings per {name}", **extra), width="stretch")

# =========================
# Word analysis
# =========================
st.divider()
st.header("📅 Historical Word Analysis: Track the Evolution")

available_years = sorted(df_time["year"].dropna().astype(int).astype(str).unique(), reverse=True)
selected_year = st.selectbox("Select a year to analyze the feedback:", options=["All Years"] + available_years)
df_words = df_time if selected_year == "All Years" else df_time[df_time["year"] == int(selected_year)]


def top_words(texts, n=15):
    words = " ".join(texts.astype(str)).lower().split()
    counts = Counter(w for w in words if w not in CUSTOM_STOPWORDS and len(w) > 2)
    return pd.DataFrame(counts.most_common(n), columns=["Word", "Count"])


col7, col8 = st.columns(2)
for col, label, mask, color, empty_text in [
    (col7, "✅ Positive Insights", df_words["rating"] == 5, "#2ecc71", "No 5-star reviews"),
    (col8, "❌ Negative Insights", df_words["rating"] <= 2, "#e74c3c", "No negative reviews"),
]:
    with col:
        st.subheader(f"{label} ({selected_year})")
        texts = df_words.loc[mask, "review_text_clean_advanced"].dropna()
        if texts.empty:
            st.info(f"{empty_text} found for {selected_year}.")
            continue
        fig = px.bar(top_words(texts), x="Count", y="Word", orientation="h", color_discrete_sequence=[color])
        fig.update_layout(yaxis={"categoryorder": "total ascending"}, height=450)
        st.plotly_chart(fig, width="stretch")

st.divider()
st.header("Final Analysis: The Evolution of Customer Sentiment")
with st.container(border=True):
    st.markdown("""
### 📉 Strategic Insights: 2018 vs. 2025

Our data-driven journey reveals a significant shift in customer experience, confirming the **"Logistics-Collapse-Theory"** following the post-pandemic e-commerce boom:

- **2018 - The "Niche Expert" Era:** High satisfaction driven by competitive pricing and a reliable niche service. Negative feedback was low-volume and mostly centered around general support issues.
- **2022-2025 - The "Mass Market Stress Test":** As order volumes exploded (confirming the online-shift theory), the shop's logistics reached a breaking point.
- **The "Tire Crisis":** By 2025, **'Tire'** emerged as the #1 driver of negative reviews. Bulky goods logistics seem unable to keep up with the scale of the business.
- **Time as a Currency:** Words like *'waiting'*, *'week'*, and *'never'* dominate 1-star reviews in 2025, showing that delivery delays have moved from "minor inconvenience" to a "trust-killing" factor.
- **The Polarization:** Interestingly, *'delivery'* remains a top word in 5-star reviews too. This indicates a "hit-or-miss" logistics system: It's either excellent or a total failure, with very little middle ground.
""")
    st.error("⚠️ **Conclusion:** To stabilize the rating in 2026, the company must urgently fix its bulky goods "
             "(Tire) fulfillment and refund communication.")

# =========================
# Feature selection
# =========================
st.info("💡 **Feature Selection Update:** Time features (year, month_name, weekday, season, day_period) will be "
        "removed from the dataset because they are not directly relevant for our current analysis of customer reviews.")
dropped = list(TIME_COLS)
shown_cols = [c for c in display_cols if c != "date"]
status_table(shown_cols, df, cleaned=["review_text"], dropped=dropped)
st.divider()

# --- Location ---
top_locations = df["location"].value_counts(dropna=False).head(15).rename_axis("Location").reset_index(name="Count")
top_locations["Location"] = top_locations["Location"].fillna("Unknown")
fig = px.bar(top_locations, x="Location", y="Count", text="Count", color_discrete_sequence=["#636EFA"],
             title="📍 Top 15 Review Locations (including Unknown)")
fig.update_layout(xaxis_tickangle=-45, font=dict(size=14), height=600, xaxis_title="City / Location",
                  yaxis_title="Number of Reviews", template="plotly_white", showlegend=False)
st.plotly_chart(fig, width="stretch")

top_locs = df["location"].value_counts().head(10).index
loc_ratings = df[df["location"].isin(top_locs)].groupby(["location", "rating"]).size().unstack(fill_value=0)
loc_ratings = loc_ratings.rename(columns=lambda r: RATING_LABELS[int(r) - 1])
st.plotly_chart(heatmap(row_percent(loc_ratings), "Rating (Stars)", "Location", "Percentage %", "RdYlGn", ".1f",
                        "🎯 Detailed Rating Distribution per Location", font=dict(size=14)), width="stretch")

st.success("✅ Column 'location' was successfully dropped.")
dropped.append("location")
shown_cols.remove("location")
status_table(shown_cols, df, cleaned=["review_text"], dropped=dropped)
st.divider()

# --- Verified ---
st.header("🛡️ Verified Status vs. Rating Distribution")
verified_counts = df.groupby(["verified", "rating"]).size().reset_index(name="count")
verified_counts["status"] = verified_counts["verified"].map({0: "Not Verified (0)", 1: "Verified (1)"})
fig_ver = px.bar(verified_counts, x="status", y="count", color="rating", barmode="relative",
                 title="🛡️ Verified Status vs. Rating Distribution", color_continuous_scale="RdYlGn",
                 labels={"count": "Number of Reviews", "status": "Verification Status", "rating": "Stars"})
fig_ver.update_layout(font=dict(size=14), xaxis_title="", yaxis_title="Count of Reviews", legend_title="Rating")
st.plotly_chart(fig_ver, width="stretch")

avg_by_verified = df.groupby("verified")["rating"].mean()
st.info(f"""
💡 **Quick Stats:**
* Average Rating (Verified): **{avg_by_verified.get(1, float('nan')):.2f} ⭐**
* Average Rating (Not Verified): **{avg_by_verified.get(0, float('nan')):.2f} ⭐**
""")

st.divider()
ver_ratings = df.groupby(["verified", "rating"]).size().unstack(fill_value=0)
ver_ratings.index = ver_ratings.index.map({0: "Not Verified (0)", 1: "Verified (1)"})
ver_ratings = ver_ratings.rename(columns=lambda r: RATING_LABELS[int(r) - 1])
small = dict(height=450, margin=dict(l=20, r=20, t=50, b=20))
col_v1, col_v2 = st.columns(2)
col_v1.plotly_chart(heatmap(row_percent(ver_ratings), "Rating (Stars)", "Verification Status", "Percentage %",
                            "RdYlGn", ".1f", "🎯 Relative Distribution (%)", **small), width="stretch")
col_v2.plotly_chart(heatmap(ver_ratings, "Rating (Stars)", "Verification Status", "Count", "Blues", True,
                            "📊 Absolute Counts", **small), width="stretch")

st.info("""
💡 **Feature Selection:** 'verified' stays.

💡 However, the 'company' column will be removed, as it shows no significant correlation with the rating and could introduce model bias.
""")
dropped.append("company")
shown_cols.remove("company")
status_table(shown_cols, df, cleaned=["review_text", "verified"], dropped=dropped)

# --- Supplier response ---
st.divider()
st.header("🎯 Supplier Response Analysis")
df["has_response"] = df["supplier_response"].notna().astype(int)
resp_ratings = df.groupby(["has_response", "rating"]).size().unstack(fill_value=0)
resp_ratings.index = resp_ratings.index.map({0: "No Response (0)", 1: "Has Response (1)"})
resp_ratings = resp_ratings.rename(columns=lambda r: RATING_LABELS[int(r) - 1])
col_r1, col_r2 = st.columns(2)
col_r1.plotly_chart(heatmap(row_percent(resp_ratings), "Rating (Stars)", "Supplier Response Status", "Percentage %",
                            "RdYlGn", ".1f", "🎯 Relative Influence (%)", **small), width="stretch")
col_r2.plotly_chart(heatmap(resp_ratings, "Rating (Stars)", "Supplier Response Status", "Count", "Blues", True,
                            "📊 Absolute Counts", **small), width="stretch")

avg_by_response = df.groupby("has_response")["rating"].mean()
st.info(f"""
💡 **Quick Insight:**
* Average Rating with Response: **{avg_by_response.get(1, float('nan')):.2f} ⭐**
* Average Rating without Response: **{avg_by_response.get(0, float('nan')):.2f} ⭐**
""")
st.info("""
💡 **Business Intelligence Insight:**
Companies show a strong reactive pattern: they prioritize responding to **negative reviews** (1-star)
as part of crisis management, while positive feedback often remains unacknowledged.
To avoid 'Data Leakage' in our prediction model, this post-event feature will now be removed.
""")

reply_rate = row_percent(df.groupby(["verified", "has_response"]).size().unstack(fill_value=0))
reply_rate.index = reply_rate.index.map({0: "Not Verified (0)", 1: "Verified (1)"})
reply_rate.columns = reply_rate.columns.map({0: "No Response", 1: "Has Response"})
st.plotly_chart(heatmap(reply_rate, "Company Reaction", "Customer Status", "Percentage %", "Purples", ".1f",
                        "🛡️ Response Strategy: Do Companies care more about Verified Customers?"), width="stretch")
st.info(f"""
📊 **Result:**
* **{reply_rate.loc['Verified (1)', 'Has Response']:.1f}%** of verified customers got a reply.
* **{reply_rate.loc['Not Verified (0)', 'Has Response']:.1f}%** of unverified customers got a reply.
""")

dropped.append("supplier_response")
shown_cols.remove("supplier_response")
status_table(shown_cols, df, cleaned=["review_text", "verified"], dropped=dropped)

# =========================
# Final dataset
# =========================
st.divider()
df_final = df[["verified", "review_text", "rating"]]
st.write("### 🏆 Final Processed Dataset (Top 15 Rows)")
st.dataframe(df_final.head(15), width="stretch")
st.success(f"🏁 **Phase 'Preprocessing' Complete:** Our dataset is now high-octane fuel for Machine Learning!  \n"
           f"Final Model-Ready Shape: **{df_final.shape[0]}** reviews and **{df_final.shape[1]}** core features.  \n"
           f"*✅ {len(df_final)} unique reviews ready for Modelling!*")
st.page_link("pages/03_Feature_Engineering.py", label="Continue to Feature Engineering", icon="➡️")
