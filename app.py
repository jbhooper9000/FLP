import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go 
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
df['office'] = df['office'].replace({'brtn': 'Brighton', 'hors': 'Horsham', 'lndn': 'London', 'othr': 'Other'})
df['acting_solicitor'] = df['acting_solicitor'].str.upper()

# --- SIDEBAR ---

st.sidebar.image("Logo-2023.png", use_column_width=True)
st.sidebar.header("Filters:")



def sidebarfilter(df, filter, label):
  
  container = st.sidebar.expander(label)
  all = container.checkbox("Select all", value=True, key=label)
  if all:
      box = container.multiselect('', options= df[filter].unique(),
                                   default= df[filter].unique(),
                                  key = filter
                                    )
  else:
      box =  container.multiselect('', options= df[filter].unique(),
                                   key = filter
                                      )
  return box

office = sidebarfilter(df, 'office', '**Location**')
df_office = df.query('office == @office')

acting_solicitor = sidebarfilter(df_office, 'acting_solicitor', '**Solicitor**')
df_solicitor = df_office.query('acting_solicitor == @acting_solicitor')

case_type = sidebarfilter(df_solicitor, 'case_type', '**Case Type**')
dr_used = sidebarfilter(df_solicitor, 'dr_used', '**DR Used**')
children = sidebarfilter(df_solicitor, 'children', '**Children**')
family_home = sidebarfilter(df_solicitor, 'family_home', '**Family Home**')
partner_solicitor = sidebarfilter(df_solicitor, 'partner_solicitor', '**Partner Solicitor**')


df_selection = df_solicitor.query(
               'case_type == @case_type & dr_used == @dr_used & children == @children &  family_home == @family_home & partner_solicitor == @partner_solicitor'
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
fig.patch.set_facecolor('#ffffff')

r_col.pyplot(fig)

st.markdown('##')

# --- STATS ---

if df_selection.shape[0] != 0:
    median_lifetime_value = df_selection.lifetime_value.median()
    median_case_duration = int(df_selection.case_duration.median())
    median_case_hours = df_selection.case_hours.median()
    avg_conflict = df_selection.cooperation.mean()
else:
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


#--  Time Series Function --#

st.subheader('Time Series Plot')

def seasonaldf(df, column=None, agg='count'):
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    if column is None:
        p = pd.DataFrame(df['start_date'])
        p['month'] = p['start_date'].dt.month_name().str[:3]
        p['year'] = p['start_date'].dt.year
        p['values'] = 1
        p = p.drop(['start_date'], axis=1)
        p = p.pivot_table(index='month', columns='year', values='values', aggfunc=agg)
        p = p.reindex(months)
    else:
        p = df[['start_date', column]]
        p['month'] = p['start_date'].dt.month_name().str[:3]
        p['year'] = p['start_date'].dt.year
        p = p.drop(['start_date'], axis=1)
        p = p.pivot_table(index='month', columns='year', values=column, aggfunc=agg)
        p = p.reindex(months)
    return p

p1 = seasonaldf(df_selection)
p2 = seasonaldf(df_selection, 'lifetime_value', np.sum)
p3 = seasonaldf(df_selection, 'case_hours', np.sum)
p4 = seasonaldf(df_selection, 'case_duration', np.sum)
p5 = seasonaldf(df_selection, 'billings', np.sum)

p_all = pd.concat([p1,p2,p3,p4,p5], axis=1)


# --- TIME SERIES PLOT --- #

fig_ts = go.Figure()
mask = []
if df_selection.shape[0] != 0:
  for i in range(int(len(p_all.columns)/len(p1.columns))):
    for j in range(i*len(p1.columns),len(p1.columns)+i*len(p1.columns)):
      if i == 0:
        fig_ts.add_trace(go.Scatter(
                        x = p_all.index,
                        y = p_all.iloc[:,j],
                        name = p_all.columns[j].astype(str),
                        visible = True
                        )
          )
        mask.append(i)
      else:
        fig_ts.add_trace(go.Scatter(
                        x = p_all.index,
                        y = p_all.iloc[:,j],
                        name = p_all.columns[j].astype(str),
                        visible = False
                        )
          )
        mask.append(i)
  fig_ts.update_layout(title='Number of Cases',
      updatemenus=[go.layout.Updatemenu(
          active=0,
          buttons=list(
              [dict(label = 'Number of Cases',
                    method = 'update',
                    args = [{'visible': [x == 0 for x in mask]}, # the index of True aligns with the indices of plot traces
                            {'title': 'Number of Cases',
                             'showlegend':True}]),

               dict(label = 'Lifetime Value',
                    method = 'update',
                    args = [{'visible': [x == 1 for x in mask]},
                            {'title': 'Lifetime Value',
                             'showlegend':True}]),

               dict(label = 'Case Hours',
                    method = 'update',
                    args = [{'visible': [x == 2 for x in mask]}, # the index of True aligns with the indices of plot traces
                            {'title': 'Case Hours',
                             'showlegend':True}]),

               dict(label = 'Case Duration',
                    method = 'update',
                    args = [{'visible': [x == 3 for x in mask]},
                            {'title': 'Case Duration',
                             'showlegend':True}]),

               dict(label = 'Billings',
                    method = 'update',
                    args = [{'visible': [x == 4 for x in mask]}, # the index of True aligns with the indices of plot traces
                            {'title': 'Billings',
                             'showlegend':True}])
              ])
          )
      ])
else:
  pass
st.plotly_chart(fig_ts)

st.markdown('---')


#--  Pie Plot --#

st.subheader('Pie Plot')

df_pie = df.query(
               'case_type == @case_type & dr_used == @dr_used & children == @children &  family_home == @family_home & partner_solicitor == @partner_solicitor'
    )
df_pie = df_pie.dropna(subset=['office'])

df_pie = df_pie[['office','lifetime_value','case_hours','case_duration','billings']]

fig_pie = go.Figure()
pimask = []
for i in range(len(df_pie.columns)):
        if i == 0:
            fig_pie.add_trace(go.Pie(
                      labels = df_pie['office'],
                      visible = True
                      ))
        else:
            fig_pie.add_trace(go.Pie(
                      values = df_pie[df_pie.columns[i]],
                      labels = df_pie['office'],
                      visible = True
                      ))
        pimask.append(i)


fig_pie.update_layout(
    updatemenus=[go.layout.Updatemenu(
        active=0,
        buttons=list(
            [dict(label = 'Number of Cases',
                  method = 'update',
                  args = [{'visible': [x == 0 for x in mask]}, # the index of True aligns with the indices of plot traces
                          {'title': 'Number of Cases',
                           'showlegend':True}]),
             
             dict(label = 'Lifetime Value',
                  method = 'update',
                  args = [{'visible': [x == 1 for x in mask]},
                          {'title': 'Lifetime Value',
                           'showlegend':True}]),
             
             dict(label = 'Case Hours',
                  method = 'update',
                  args = [{'visible': [x == 2 for x in mask]}, # the index of True aligns with the indices of plot traces
                          {'title': 'Case Hours',
                           'showlegend':True}]),
             
             dict(label = 'Case Duration',
                  method = 'update',
                  args = [{'visible': [x == 3 for x in mask]},
                          {'title': 'Case Duration',
                           'showlegend':True}]),
             
             dict(label = 'Billings',
                  method = 'update',
                  args = [{'visible': [x == 4 for x in mask]}, # the index of True aligns with the indices of plot traces
                          {'title': 'Billings',
                           'showlegend':True}])
             
             
            ])
        )
    ])

st.plotly_chart(fig_pie)

st.markdown('---')

# Dataframe

st.subheader('DataFrame')

st.dataframe(df_selection)
  
# --- HIDE Streamlit STYLE

hide_style = """
           <style>
            MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_style, unsafe_allow_html=True)
