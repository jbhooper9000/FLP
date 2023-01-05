import pandas as pd
import plotly.express as px
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib as mpl


st.set_page_config(page_title ='Family Law Partners - Dashboard',
                   page_icon =':chart_with_upwards_trend:',
                   layout = 'wide')

@st.cache
def getdf():
    df = pd.read_excel('https://github.com/jbhooper9000/FLP/blob/main/PyTransform.xlsx?raw=true')
    return df

df=getdf()

# --- SIDEBAR ---

st.sidebar.image("Logo-2023.png", use_column_width=True)
st.sidebar.header("Filters:")

def sidebarfilter(filter, label):
  
  container = st.sidebar.container()
  all = st.sidebar.checkbox("Select all", value=True, key=label)
 
  if all:
      box = container.multiselect(**label**,
                                   options= df[filter].unique(),
                                   default= df[filter].unique()
                                    )
  else:
      box =  container.multiselect(**label**,
                                      options= df[filter].unique()
                                      )
  return box

office = sidebarfilter('office', 'Location')
case_type = sidebarfilter('case_type', 'Case Type')
dr_used = sidebarfilter('dr_used', 'DR Used')
children = sidebarfilter('children', 'Children')
family_home = sidebarfilter('family_home', 'Family Home')
partner_solicitor = sidebarfilter('partner_solicitor', 'Partner Solicitor')


df_selection = df.query(
    "office == @office & case_type == @case_type & dr_used == @dr_used & children == @children &  family_home == @family_home & partner_solicitor == @partner_solicitor"
    )


# --- MAINPAGE ---



st.title('Family Law Partners - Dashboard')

l_col, r_col = st.columns(2)
num_cases = df_selection.shape[0]
l_col.subheader(f'Number of Cases: {num_cases}')

# Avg Cooperation Level Plot

fig, ax = plt.subplots(figsize=(8, 1))
fig.subplots_adjust(bottom=0.5)

cmap = mpl.cm.RdYlGn
norm = mpl.colors.Normalize(vmin=1, vmax=5)

cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
             cax=ax, orientation='horizontal')

cbar.set_ticks([1,2,3,4,5], labels=['Implacable Hostility (1)', 'Low (2)', 'Medium (3)',
                                    'High (4)', 'Amicable (5)'])
cbar.ax.axvline(df_selection['cooperation'].mean(), color='#636EFA', lw=5, ls=':', dashes=(0.6, 0.6)) 
cbar.ax.tick_params(labelsize=12) 
plt.title(f'Average Client/Partner Cooperation Level: {round(df_selection.cooperation.mean(),2)}')
fig.patch.set_facecolor('#F7F5F3')

r_col.pyplot(fig)

st.markdown('##')

# --- STATS ---

if df_selection.shape[0] != 0:
    median_lifetime_value = df_selection.lifetime_value.median()
    median_case_duration = int(df_selection.case_duration.median())
    median_case_hours = df_selection.case_hours.median()
    avg_conflict = df_selection.cooperation.mean()
else :
    median_lifetime_value = median_case_duration = median_case_hours = avg_conflict = 0


# --- PLOTS ---

fig_lifetimevalue_dist = px.histogram(df_selection, x='lifetime_value', nbins=50)
fig_lifetimevalue_dist.update_layout(plot_bgcolor='#ffffff',
                               xaxis=(dict(showgrid=False)),
                               xaxis_title="Lifetime Value (£)")

fig_casehours_dist = px.histogram(df_selection, x='case_hours', nbins=50)
fig_casehours_dist.update_layout(plot_bgcolor='#ffffff',
                               xaxis=(dict(showgrid=False)),
                               xaxis_title="Case Hours")

fig_casedur_dist = px.histogram(df_selection, x='case_duration', nbins=50)
fig_casedur_dist.update_layout(plot_bgcolor='#ffffff',
                               xaxis=(dict(showgrid=False)),
                               xaxis_title="Case Duration (Months)")


left_col, mid_col, right_col = st.columns(3)
with left_col:
    st.subheader("Median Lifetime Value:")
    st.subheader(f'£ {median_lifetime_value:,.2f}')
    st.plotly_chart(fig_lifetimevalue_dist, use_container_width=True)
with mid_col:
    st.subheader('Median Case Hours:')
    st.subheader(f'{int(median_case_hours)} hours {round(((median_case_hours % 1)*60)/5)*5} minutes')
    st.plotly_chart(fig_casehours_dist, use_container_width=True)
with right_col:
    st.subheader('Median Case Duration (Months):')
    st.subheader(f'{median_case_duration}')
    st.plotly_chart(fig_casedur_dist, use_container_width=True)
    
st.markdown('---')

# mat_type = df_selection.groupby(by=['communication'])['communication'].count()

# fig_mattertype = px.bar(mat_type, orientation='h')
# fig_mattertype.update_layout(plot_bgcolor='#ffffff',
#                                 xaxis=(dict(showgrid=False)),
#                                 yaxis_title="Matter Type",
#                                 xaxis_title="Number of Cases",
#                                 showlegend=False)



# st.plotly_chart(fig_mattertype)



# st.plotly_chart(fig_conflict)

# --- HIDE Streamlit STYLE

hide_style = """
           <style>
            MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_style, unsafe_allow_html=True)
