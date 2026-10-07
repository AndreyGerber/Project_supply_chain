import pandas as pd
import plotly.express as px
import streamlit as st

from _utils import load_processed

st.set_page_config(page_title="Auto parts store Review Dashboard", layout="wide")

df = load_processed()
if df.empty:
    st.warning("Data could not be loaded. Please check the source file.")
    st.stop()

# --- Sidebar filter ---
st.sidebar.header("Filter Options")
ratings = sorted(df["rating"].unique())
selected_rating = st.sidebar.multiselect("Select Rating", options=ratings, default=ratings)
df_filtered = df[df["rating"].isin(selected_rating)].copy()

BIG_FONT = dict(
    font=dict(size=14),
    xaxis=dict(title_font=dict(size=20), tickfont=dict(size=14)),
    yaxis=dict(title_font=dict(size=20), tickfont=dict(size=14), showgrid=True, gridcolor="LightGray"),
)

# --- Header ---
st.title("📊 Phase 1: Data Exploration")
st.info("🚀 The objective of this part is to gain a first insight into the statistics of the data.")
st.markdown("""
The scraper iterates across multiple companies and pages, extracting the following attributes for each review:

- **review_text:** customer comment
- **rating_svg:** star rating
- **date:** timestamp of the review
- **location:** customer country
- **supplier_response:** company reply
- **verified:** review verification status
- **company:** retailer identifier

The initial analytics are presented below — enjoy exploring!
""")
st.divider()

# --- Raw data preview ---
st.subheader("📄 Raw Data Preview")
st.info("Direct preview of the filtered dataset:")
technical_cols = ["review_text_clean_advanced", "review_text_clean", "issue_categories",
                  "review_text_clean_light", "review_length", "sentiment", "has_negation"]
st.dataframe(df_filtered.drop(columns=technical_cols, errors="ignore").head(76), width="stretch", height=450)

# --- Company distribution ---
with st.container(border=True):
    st.markdown("#### 🏢 Company Distribution")
    company_counts = df_filtered["company"].value_counts().reset_index()
    company_counts.columns = ["Company Name", "Review Count"]
    c1, c2 = st.columns([1, 2])
    c1.dataframe(company_counts, width="stretch", hide_index=True, height=650)
    fig_comp = px.bar(company_counts, x="Review Count", y="Company Name", orientation="h",
                      height=650, title="Comments per Company")
    fig_comp.update_layout(margin=dict(l=0, r=0, t=40, b=0), yaxis={"categoryorder": "total ascending"})
    c2.plotly_chart(fig_comp, width="stretch")

# --- Rating vs verified ---
with st.container(border=True):
    st.markdown("#### 📊 Rating vs Verified")
    fig_ver = px.violin(df, x="verified", y="rating", color="verified", box=True, points="all",
                        color_discrete_map={0: "#EF553B", 1: "#00CC96"})
    counts = df.groupby(["verified", "rating"]).size().reset_index(name="count")
    for row in counts.itertuples():
        fig_ver.add_annotation(x=row.verified, y=row.rating, text=f"n={row.count}", showarrow=False,
                               yshift=10, font=dict(size=12, color="black", family="Arial"))
    fig_ver.update_layout(xaxis=dict(tickmode="array", tickvals=[0, 1],
                                     ticktext=["Non-Verified (0)", "Verified (1)"]))
    st.plotly_chart(fig_ver, width="stretch")
    st.markdown("""
    The inclusion of a **'verified'** indicator allows us to distinguish
    between authenticated and non-authenticated customer feedback,
    reducing potential bias and increasing the reliability of the analysis.
    """)

# --- Analysis period & KPIs ---
st.markdown("#### 📅 Analysis Period")
if not df_filtered.empty:
    first_date, last_date = df_filtered["date"].min(), df_filtered["date"].max()
    st.success(f"✅ This dataset covers reviews from **{first_date:%d.%m.%Y}** to **{last_date:%d.%m.%Y}**.")

    timeline_df = pd.DataFrame({"date": [first_date, last_date],
                                "label": ["first comment", "last comment"], "y": [0, 0]})
    fig_timeline = px.line(timeline_df, x="date", y="y", markers=True, text="label")
    fig_timeline.update_traces(line_color="#2E7D32", line_width=4, marker=dict(size=12, symbol="diamond"),
                               textposition="top center", textfont=dict(size=16, weight="bold"))
    fig_timeline.update_layout(height=120, margin=dict(l=20, r=20, t=30, b=20), plot_bgcolor="rgba(0,0,0,0)",
                               xaxis=dict(showgrid=False, title=""),
                               yaxis=dict(showgrid=False, showticklabels=False, title=""))
    st.plotly_chart(fig_timeline, width="stretch", config={"displayModeBar": False})

    with st.container(border=True):
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Reviews", len(df_filtered))
        col2.metric("Average Rating", f"{df_filtered['rating'].mean():.2f} / 5.0")
        col3.metric("Supplier Response Rate", f"{df_filtered['supplier_response'].notna().mean() * 100:.1f}%")

# --- Comments per year ---
with st.container(border=True):
    st.markdown("#### 📊 Number of comments by Year")
    yearly_counts = df_filtered["date"].dt.year.astype("Int64").astype(str).value_counts().sort_index().reset_index()
    yearly_counts.columns = ["Year", "Number of comments"]
    fig_years = px.bar(yearly_counts, x="Year", y="Number of comments", text="Number of comments", color="Year",
                       color_discrete_sequence=["#1E88E5"] * len(yearly_counts), height=600)
    fig_years.update_layout(**BIG_FONT, xaxis_type="category", plot_bgcolor="rgba(0,0,0,0)",
                            legend=dict(title="Select Year:", yanchor="top", y=1, xanchor="left", x=1.02),
                            margin=dict(r=150))
    fig_years.update_traces(textposition="outside")
    st.plotly_chart(fig_years, width="stretch")

st.divider()

# --- Analysis tabs ---
tab1, tab2, tab3 = st.tabs(["📈 Performance Trends", "💬 Feedback Analysis", "📍 Operations & Support"])

with tab1:
    st.subheader("Customer Satisfaction Distribution")
    fig = px.histogram(df_filtered, x="rating", color="rating", title="Frequency of Ratings", nbins=5, height=600,
                       labels={"rating": "Star Rating", "count": "Number of Comments"},
                       color_discrete_map={1: "#2E7D32", 2: "#311B92", 3: "#FBC02D", 4: "#81D4FA", 5: "#C62828"})
    fig.update_layout(**BIG_FONT)
    fig.update_yaxes(title="Count of Star Ratings")
    st.plotly_chart(fig, width="stretch")

with tab2:
    st.subheader("📈 Average rating per company over the years")
    min_reviews = st.slider("min number of comments per year:", min_value=5, max_value=15, value=7)

    df_time = df_filtered.assign(year=df_filtered["date"].dt.year)
    min_year, max_year = int(df_time["year"].min()), pd.Timestamp.now().year
    df_trend = (df_time.groupby(["year", "company"])
                .agg(avg_rating=("rating", "mean"), review_count=("rating", "count"))
                .reset_index())
    df_trend = df_trend[df_trend["review_count"] >= min_reviews]

    companies = df_trend["company"].unique()
    if len(companies) > 0:
        # Every year appears on the x-axis, missing years stay empty instead of dropping to zero
        full_index = pd.MultiIndex.from_product([range(min_year, max_year + 1), companies], names=["year", "company"])
        df_trend = df_trend.set_index(["year", "company"]).reindex(full_index).reset_index()

        fig_trend = px.line(df_trend, x="year", y="avg_rating", color="company", markers=True,
                            title=f"Trends in Customer Satisfaction ({min_year} - {max_year})",
                            labels={"year": "Year", "avg_rating": "Ø Stars", "company": "Company"},
                            hover_data={"review_count": True}, height=600,
                            color_discrete_sequence=px.colors.qualitative.Safe)
        fig_trend.update_layout(
            title_font=dict(size=22),
            xaxis=dict(tickmode="linear", dtick=1, range=[min_year - 0.1, max_year + 0.1],
                       title_font=dict(size=20), tickfont=dict(size=14), gridcolor="rgba(200, 200, 200, 0.3)"),
            yaxis=dict(range=[0.8, 5.2], dtick=1, title="Rating (Ø Stars)",
                       title_font=dict(size=20), tickfont=dict(size=14)),
            legend=dict(font=dict(size=14), yanchor="top", y=1, xanchor="left", x=1.02),
            margin=dict(l=60, r=150, t=80, b=60), hovermode="x unified", plot_bgcolor="white",
        )
        fig_trend.update_traces(connectgaps=False, line=dict(width=3))
        st.plotly_chart(fig_trend, width="stretch")
    else:
        st.info("Lower the filter or select more ratings to see data.")

with tab3:
    st.header("📍 Geographic & Support Performance")
    col_a, col_b = st.columns([7, 3])

    with col_a:
        loc_counts = df_filtered["location"].value_counts()
        if len(loc_counts) > 9:
            loc_counts = pd.concat([loc_counts.head(9), pd.Series({"Others": loc_counts.iloc[9:].sum()})])
        plot_df = loc_counts.rename_axis("Region").reset_index(name="Count")
        fig_loc = px.bar(plot_df, x="Region", y="Count", title="Top 9 Regions & Others", text="Count",
                         color="Region", color_discrete_sequence=px.colors.qualitative.Pastel, height=500)
        fig_loc.update_layout(showlegend=False, yaxis_title="Number of Reviews",
                              xaxis={"categoryorder": "total descending"}, margin=dict(t=50, b=50, l=20, r=20))
        fig_loc.update_traces(textposition="outside")
        st.plotly_chart(fig_loc, width="stretch")

    with col_b:
        resp_counts = (df_filtered["supplier_response"].notna().value_counts()
                       .rename({True: "Responded", False: "Pending"}))
        fig_resp = px.bar(x=resp_counts.index, y=resp_counts.values, title="Response Status",
                          color=resp_counts.index, height=500,
                          color_discrete_map={"Responded": "#2E6AD1", "Pending": "#89C6FF"})
        fig_resp.update_layout(showlegend=False, xaxis_title=None, yaxis_title="Count",
                               margin=dict(t=50, b=50, l=20, r=20))
        fig_resp.update_traces(texttemplate="%{y}", textposition="outside")
        st.plotly_chart(fig_resp, width="stretch")

# --- Footer ---
st.divider()
st.markdown("<h2 style='text-align: center; color: #ff4b4b;'>Thank you for exploring the Autodoc Review Dashboard!</h2>",
            unsafe_allow_html=True)

with st.container(border=True):
    st.markdown("#### Next Steps will be:")
    st.page_link("pages/02_Preprocessing.py", label="**Preprocessing** → preparing the data for machine learning models")
    st.page_link("pages/04_Modelling.py", label="**Modelling** → analyzing the data with various machine learning "
                                                "techniques and choosing the best model for our use case.")
    st.page_link("pages/05_Live_Demo.py", label="**Play with our model** → testing the model with new comments "
                                                "and evaluating the results.")
