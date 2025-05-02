import streamlit as st
import pandas as pd
import pulp
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO
import numpy as np
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="Tối ưu Logistics", layout="wide")

st.title("Ứng dụng Tối ưu hoá chi phí Logistics doanh nghiệp Dệt May")

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
        
        # Tính toán cho cả hai tuyến đường
        for vol in volume_range:
            # Tính cho Nội Bài
            prob_nb = pulp.LpProblem("Sensitivity_Analysis_NB", pulp.LpMinimize)
            vehicle_vars_nb = {v: pulp.LpVariable(f"{v}_nb", 0, None, pulp.LpInteger) 
                           for v in vehicles.keys()}
            
            prob_nb += pulp.lpSum(vehicles[v]['cost_nb'] * vehicle_vars_nb[v] for v in vehicles.keys())
            prob_nb += pulp.lpSum(vehicles[v]['capacity'] * vehicle_vars_nb[v] for v in vehicles.keys()) >= vol
            prob_nb.solve()
            
            total_cost_nb = sum(vehicle_vars_nb[v].value() * vehicles[v]['cost_nb'] 
                            for v in vehicles.keys())
            
            # Tính cho Hải Phòng
            prob_hp = pulp.LpProblem("Sensitivity_Analysis_HP", pulp.LpMinimize)
            vehicle_vars_hp = {v: pulp.LpVariable(f"{v}_hp", 0, None, pulp.LpInteger) 
                           for v in vehicles.keys()}
            
            prob_hp += pulp.lpSum(vehicles[v]['cost_hp'] * vehicle_vars_hp[v] for v in vehicles.keys())
            prob_hp += pulp.lpSum(vehicles[v]['capacity'] * vehicle_vars_hp[v] for v in vehicles.keys()) >= vol
            prob_hp.solve()
            
            total_cost_hp = sum(vehicle_vars_hp[v].value() * vehicles[v]['cost_hp'] 
                            for v in vehicles.keys())
            
            sensitivity_results.append({
                'Thể tích': vol, 
                'Chi phí Nội Bài': total_cost_nb,
                'Chi phí Hải Phòng': total_cost_hp
            })
        
        df_sensitivity = pd.DataFrame(sensitivity_results)
        
        # Vẽ biểu đồ độ nhạy cho cả hai tuyến đường
        fig_sensitivity = go.Figure()
        
        # Thêm đường cho Nội Bài
        fig_sensitivity.add_trace(go.Scatter(
            x=df_sensitivity['Thể tích'],
            y=df_sensitivity['Chi phí Nội Bài']/1000000,  # Chuyển đổi sang đơn vị triệu
            name='Nội Bài',
            line=dict(color='rgb(0, 102, 204)')
        ))
        
        # Thêm đường cho Hải Phòng
        fig_sensitivity.add_trace(go.Scatter(
            x=df_sensitivity['Thể tích'],
            y=df_sensitivity['Chi phí Hải Phòng']/1000000,  # Chuyển đổi sang đơn vị triệu
            name='Hải Phòng',
            line=dict(color='rgb(255, 102, 102)')
        ))
        
        fig_sensitivity.update_layout(
            title='Phân tích độ nhạy chi phí theo thể tích',
            xaxis_title='Thể tích (m³)',
            yaxis_title='Chi phí (triệu VNĐ)',
            hovermode='x unified'
        )
        
        st.plotly_chart(fig_sensitivity)
        
        # Hiển thị bảng dữ liệu
        st.subheader("Bảng dữ liệu chi tiết")
        display_df = df_sensitivity.copy()
        display_df['Chi phí Nội Bài'] = display_df['Chi phí Nội Bài'].apply(lambda x: f"{x:,.0f} VNĐ")
        display_df['Chi phí Hải Phòng'] = display_df['Chi phí Hải Phòng'].apply(lambda x: f"{x:,.0f} VNĐ")
        st.dataframe(display_df)
        
        # Tính và hiển thị chênh lệch chi phí
        st.subheader("Phân tích chênh lệch")
        avg_diff = (df_sensitivity['Chi phí Hải Phòng'] - df_sensitivity['Chi phí Nội Bài']).mean()
        max_diff = (df_sensitivity['Chi phí Hải Phòng'] - df_sensitivity['Chi phí Nội Bài']).max()
        
        col1, col2 = st.columns(2)
        col1.metric("Chênh lệch trung bình", f"{avg_diff:,.0f} VNĐ")
        col2.metric("Chênh lệch tối đa", f"{max_diff:,.0f} VNĐ")
        
    except Exception as e:
        st.error(f"Có lỗi xảy ra trong phân tích độ nhạy: {str(e)}")
        
# Tab 3: Dự báo nhu cầu
with tab3:
    st.subheader("Dự báo nhu cầu vận chuyển 2026")
    
    try:
        # Dữ liệu thực tế 2025
        real_data = {
            'Ngày': ['2025-01-02', '2025-01-15', '2025-01-22', '2025-02-26', '2025-03-26',
                     '2025-06-25', '2025-07-16', '2025-07-30', '2025-08-06', '2025-08-13',
                     '2025-09-03', '2025-09-17', '2025-09-24', '2025-10-01'],
            'Thể tích': [8, 13, 17, 23, 18, 26, 23, 20, 18, 14, 13, 16, 6, 27],
            'Tuyến đường': ['Hải Phòng', 'Nội Bài', 'Hải Phòng', 'Nội Bài', 'Nội Bài',
                           'Hải Phòng', 'Hải Phòng', 'Hải Phòng', 'Hải Phòng', 'Hải Phòng',
                           'Hải Phòng', 'Hải Phòng', 'Hải Phòng', 'Hải Phòng']
        }
        
        df_real = pd.DataFrame(real_data)
        df_real['Ngày'] = pd.to_datetime(df_real['Ngày'])
        
        # Phân tích dữ liệu theo quý
        df_real['Quý'] = df_real['Ngày'].dt.quarter
        quarterly_avg = df_real.groupby('Quý')['Thể tích'].mean()
        
        # Hiển thị thống kê 2025
        st.subheader("Phân tích dữ liệu 2025")
        stats_col1, stats_col2, stats_col3, stats_col4 = st.columns(4)
        
        with stats_col1:
            st.metric("Q1-2025", f"{quarterly_avg[1]:.1f} m³")
        with stats_col2:
            st.metric("Q2-2025", f"{quarterly_avg[2]:.1f} m³")
        with stats_col3:
            st.metric("Q3-2025", f"{quarterly_avg[3]:.1f} m³")
        with stats_col4:
            st.metric("Q4-2025", f"{quarterly_avg[4]:.1f} m³")

        # Tùy chọn dự báo
        forecast_col1, forecast_col2 = st.columns(2)
        
        with forecast_col1:
            growth_rate = st.number_input("Tốc độ tăng trưởng năm 2026 (%)", 
                                        min_value=-20.0, max_value=20.0, value=5.0)
            
        with forecast_col2:
            seasonality = st.selectbox(
                "Mô hình theo mùa",
                ["Theo quý 2025", "Điều chỉnh theo mùa"]
            )

        # Tạo dữ liệu dự báo 2026
        future_dates = pd.date_range(start='2026-01-01', end='2026-12-31', freq='M')
        base_volume = df_real['Thể tích'].mean()
        
        forecast_volumes = []
        for date in future_dates:
            quarter = (date.month - 1) // 3 + 1
            
            if seasonality == "Theo quý 2025":
                # Sử dụng mô hình theo quý của 2025
                base = quarterly_avg[quarter]
            else:
                # Điều chỉnh theo mùa
                seasonal_factors = {
                    1: 1.1,  # Q1: Tăng do đầu năm
                    2: 0.9,  # Q2: Giảm nhẹ
                    3: 0.8,  # Q3: Thấp điểm
                    4: 1.2   # Q4: Cao điểm cuối năm
                }
                base = base_volume * seasonal_factors[quarter]
            
            # Áp dụng tăng trưởng
            volume = base * (1 + growth_rate/100)
            forecast_volumes.append(volume)

        # Tạo DataFrame dự báo
        df_forecast = pd.DataFrame({
            'Ngày': future_dates,
            'Dự báo': forecast_volumes
        })
        
        # Vẽ biểu đồ
        fig = go.Figure()
        
        # Dữ liệu thực tế 2025
        fig.add_trace(go.Scatter(
            x=df_real['Ngày'],
            y=df_real['Thể tích'],
            mode='markers+lines',
            name='Thực tế 2025',
            line=dict(color='rgb(0, 102, 204)'),
            marker=dict(size=8)
        ))
        
        # Dự báo 2026
        fig.add_trace(go.Scatter(
            x=df_forecast['Ngày'],
            y=df_forecast['Dự báo'],
            mode='lines',
            name='Dự báo 2026',
            line=dict(color='rgb(255, 102, 102)', dash='dash')
        ))
        
        fig.update_layout(
            title='So sánh dữ liệu thực tế 2025 và dự báo 2026',
            xaxis_title='Thời gian',
            yaxis_title='Thể tích (m³)',
            hovermode='x unified',
            showlegend=True
        )
        
        st.plotly_chart(fig)
        
        # Hiển thị thống kê so sánh
        st.subheader("So sánh 2025-2026")
        
        compare_col1, compare_col2, compare_col3 = st.columns(3)
        
        with compare_col1:
            st.metric(
                "Trung bình 2025",
                f"{df_real['Thể tích'].mean():.1f} m³",
                f"±{df_real['Thể tích'].std():.1f} m³"
            )
            
        with compare_col2:
            st.metric(
                "Dự báo trung bình 2026",
                f"{df_forecast['Dự báo'].mean():.1f} m³",
                f"{df_forecast['Dự báo'].mean() - df_real['Thể tích'].mean():.1f} m³"
            )
            
        with compare_col3:
            growth = (df_forecast['Dự báo'].mean() - df_real['Thể tích'].mean()) / df_real['Thể tích'].mean() * 100
            st.metric(
                "Tăng trưởng dự kiến",
                f"{growth:+.1f}%",
                "Tăng" if growth > 0 else "Giảm"
            )
        
        # Hiển thị bảng dữ liệu
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Dữ liệu thực tế 2025")
            st.dataframe(df_real[['Ngày', 'Thể tích', 'Tuyến đường']])
            
        with col2:
            st.subheader("Dự báo 2026")
            df_forecast_display = df_forecast.copy()
            df_forecast_display['Dự báo'] = df_forecast_display['Dự báo'].round(1)
            st.dataframe(df_forecast_display)
        
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
