import streamlit as st
import pandas as pd
import pulp
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO
import numpy as np
from datetime import datetime, timedelta

st.set_page_config(page_title="Tối ưu Logistics", layout="wide")

st.title("Logistics cost optimizer")

# Tabs cho các chức năng khác nhau
tab1, tab2, tab3, tab4 = st.tabs(["Tối ưu hóa", "Phân tích độ nhạy", "Dự báo nhu cầu", "Quản lý dữ liệu"])

# Định nghĩa thông tin phương tiện
vehicles = {
    'Xe 1.5 tấn': {'capacity': 10, 'cost_nb': 855000, 'cost_hp': 1950000},
    'Xe 1.9 tấn': {'capacity': 12, 'cost_nb': 1150000, 'cost_hp': 2200000},
    'Xe 3.5 tấn': {'capacity': 20, 'cost_nb': 1760000, 'cost_hp': 3780000},
    'Container 20ft': {'capacity': 35, 'cost_nb': 2680000, 'cost_hp': 5900000},
    'Container 40ft': {'capacity': 68, 'cost_nb': 2900000, 'cost_hp': 6400000}
}

# Tab 1: Tối ưu hóa
with tab1:
    with st.form("logistics_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            cargo_volume = st.number_input("Tổng thể tích hàng (m³)", min_value=0.1, value=10.0)
            route = st.selectbox("Tuyến đường", ["Nội Bài", "Hải Phòng"])
        
        submitted = st.form_submit_button("Tối ưu hóa")

    if submitted:
        try:
            # Tạo mô hình tối ưu
            prob = pulp.LpProblem("Logistics_Optimization", pulp.LpMinimize)
            
            # Biến quyết định
            vehicle_vars = {}
            for v in vehicles.keys():
                vehicle_vars[v] = pulp.LpVariable(v.replace(" ", "_"), 0, None, pulp.LpInteger)
            
            # Hàm mục tiêu
            if route == "Nội Bài":
                prob += pulp.lpSum(vehicles[v]['cost_nb'] * vehicle_vars[v] for v in vehicles.keys())
            else:
                prob += pulp.lpSum(vehicles[v]['cost_hp'] * vehicle_vars[v] for v in vehicles.keys())
            
            # Ràng buộc về thể tích
            prob += pulp.lpSum(vehicles[v]['capacity'] * vehicle_vars[v] for v in vehicles.keys()) >= cargo_volume
            
            # Giải bài toán
            prob.solve()
            
            # Hiển thị kết quả
            results = []
            total_cost = 0
            total_capacity = 0
            
            for v in vehicles.keys():
                num_vehicles = int(vehicle_vars[v].value())
                cost = num_vehicles * (vehicles[v]['cost_nb'] if route == "Nội Bài" else vehicles[v]['cost_hp'])
                capacity = num_vehicles * vehicles[v]['capacity']
                
                results.append({
                    'Phương tiện': v,
                    'Số lượng': num_vehicles,
                    'Chi phí': cost,
                    'Tổng thể tích': capacity
                })
                
                total_cost += cost
                total_capacity += capacity
            
            # Tạo DataFrame từ kết quả
            df_results = pd.DataFrame(results)
            
            # Hiển thị bảng kết quả
            st.subheader("Kết quả tối ưu")
            st.dataframe(df_results)
            
            # Hiển thị tổng hợp
            col1, col2, col3 = st.columns(3)
            col1.metric("Tổng chi phí", f"{total_cost:,.0f} VNĐ")
            col2.metric("Tổng thể tích", f"{total_capacity:.1f} m³")
            col3.metric("Thể tích yêu cầu", f"{cargo_volume:.1f} m³")
            
            # Biểu đồ phân bổ
            if not df_results.empty:
                fig = go.Figure(data=[
                    go.Bar(name='Số lượng', 
                          x=df_results['Phương tiện'].tolist(), 
                          y=df_results['Số lượng'].tolist(),
                          marker_color='rgb(0, 102, 204)'),
                    go.Bar(name='Chi phí (triệu VNĐ)', 
                          x=df_results['Phương tiện'].tolist(), 
                          y=(df_results['Chi phí']/1000000).tolist(),
                          marker_color='rgb(135, 206, 250)')
                ])

                fig.update_layout(
                    barmode='group',
                    title='Phân bổ phương tiện và chi phí',
                    yaxis_title='Số lượng / Chi phí (triệu VNĐ)',
                    showlegend=True,
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1
                    )
                )
                st.plotly_chart(fig)
            else:
                st.warning("Không có dữ liệu để hiển thị biểu đồ")
                
        except Exception as e:
            st.error(f"Có lỗi xảy ra: {str(e)}")

# Tab 2: Phân tích độ nhạy
with tab2:
    st.subheader("Phân tích độ nhạy")
    
    try:
        # Phân tích theo thể tích
        volume_range = np.linspace(10, 200, 20)
        sensitivity_results = []
        
        route_analysis = st.selectbox("Chọn tuyến đường phân tích:", ["Nội Bài", "Hải Phòng"])
        
        for vol in volume_range:
            prob = pulp.LpProblem("Sensitivity_Analysis", pulp.LpMinimize)
            vehicle_vars = {v: pulp.LpVariable(v.replace(" ", "_"), 0, None, pulp.LpInteger) 
                           for v in vehicles.keys()}
            
            if route_analysis == "Nội Bài":
                prob += pulp.lpSum(vehicles[v]['cost_nb'] * vehicle_vars[v] for v in vehicles.keys())
            else:
                prob += pulp.lpSum(vehicles[v]['cost_hp'] * vehicle_vars[v] for v in vehicles.keys())
                
            prob += pulp.lpSum(vehicles[v]['capacity'] * vehicle_vars[v] for v in vehicles.keys()) >= vol
            prob.solve()
            
            total_cost = sum(vehicle_vars[v].value() * 
                            (vehicles[v]['cost_nb'] if route_analysis == "Nội Bài" else vehicles[v]['cost_hp'])
                            for v in vehicles.keys())
            
            sensitivity_results.append({'Thể tích': vol, 'Chi phí': total_cost})
        
        df_sensitivity = pd.DataFrame(sensitivity_results)
        
        # Vẽ biểu đồ độ nhạy
        fig_sensitivity = px.line(df_sensitivity, x='Thể tích', y='Chi phí',
                                title=f'Phân tích độ nhạy chi phí theo thể tích - {route_analysis}')
        fig_sensitivity.update_layout(
            yaxis_title='Chi phí (VNĐ)',
            xaxis_title='Thể tích (m³)'
        )
        st.plotly_chart(fig_sensitivity)
        
    except Exception as e:
        st.error(f"Có lỗi xảy ra trong phân tích độ nhạy: {str(e)}")

# Tab 3: Dự báo nhu cầu
with tab3:
    st.subheader("Dự báo nhu cầu")
    
    try:
        # Tạo dữ liệu mẫu cho dự báo
        dates = pd.date_range(start='2024-01-01', periods=30, freq='D')
        volumes = np.random.normal(100, 20, 30)  # Thể tích ngẫu nhiên
        df_forecast = pd.DataFrame({'Ngày': dates, 'Thể tích': volumes})
        
        # Vẽ biểu đồ dự báo
        fig_forecast = go.Figure()
        fig_forecast.add_trace(go.Scatter(x=df_forecast['Ngày'], y=df_forecast['Thể tích'],
                                        mode='lines+markers', name='Thực tế'))
        
        # Dự báo đơn giản bằng moving average
        df_forecast['Dự báo'] = df_forecast['Thể tích'].rolling(window=7).mean()
        fig_forecast.add_trace(go.Scatter(x=df_forecast['Ngày'], y=df_forecast['Dự báo'],
                                        mode='lines', name='Dự báo'))
        
        fig_forecast.update_layout(
            title='Dự báo nhu cầu vận chuyển',
            yaxis_title='Thể tích (m³)',
            xaxis_title='Ngày'
        )
        st.plotly_chart(fig_forecast)
        
    except Exception as e:
        st.error(f"Có lỗi xảy ra trong dự báo: {str(e)}")

# Tab 4: Quản lý dữ liệu
with tab4:
    st.subheader("Quản lý dữ liệu")
    
    try:
        # Upload dữ liệu
        uploaded_file = st.file_uploader("Tải lên file Excel hoặc CSV", type=['xlsx', 'csv'])
        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith('.csv'):
                    df_upload = pd.read_csv(uploaded_file)
                else:
                    df_upload = pd.read_excel(uploaded_file)
                st.write("Dữ liệu đã tải lên:")
                st.dataframe(df_upload)
            except Exception as e:
                st.error(f"Lỗi khi đọc file: {str(e)}")
        
        # Download dữ liệu mẫu
        if st.button("Tải xuống dữ liệu mẫu"):
            try:
                buffer = BytesIO()
                with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                    # Chỉ xuất các DataFrame đã được tạo
                    if 'df_results' in locals():
                        df_results.to_excel(writer, sheet_name='Kết quả', index=False)
                    if 'df_sensitivity' in locals():
                        df_sensitivity.to_excel(writer, sheet_name='Độ nhạy', index=False)
                    if 'df_forecast' in locals():
                        df_forecast.to_excel(writer, sheet_name='Dự báo', index=False)
                
                buffer.seek(0)
                st.download_button(
                    label="Tải xuống file Excel",
                    data=buffer,
                    file_name=f"logistics_data_{datetime.now().strftime('%Y%m%d')}.xlsx",
                    mime="application/vnd.ms-excel"
                )
            except Exception as e:
                st.error(f"Lỗi khi tạo file Excel: {str(e)}")
                
    except Exception as e:
        st.error(f"Có lỗi xảy ra trong quản lý dữ liệu: {str(e)}")
