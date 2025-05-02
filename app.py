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

st.title("Ứng dụng Tối ưu Logistics Dệt May")

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
    st.subheader("Dự báo nhu cầu vận chuyển")
    
    try:
        # Tùy chọn cho dự báo
        forecast_col1, forecast_col2 = st.columns(2)
        
        with forecast_col1:
            start_date = st.date_input(
                "Ngày bắt đầu",
                datetime.now() - timedelta(days=30)
            )
            forecast_days = st.number_input("Số ngày dự báo", min_value=7, max_value=90, value=30)
            
        with forecast_col2:
            seasonality = st.selectbox(
                "Mô hình theo mùa",
                ["Không có", "Theo tuần", "Theo tháng"]
            )
            base_volume = st.number_input("Thể tích trung bình/ngày (m³)", min_value=1.0, value=100.0)

        # Tạo dữ liệu lịch sử giả lập với tính mùa vụ
        dates = pd.date_range(start=start_date, periods=30, freq='D')
        
        # Tạo dữ liệu cơ bản
        base_volumes = np.random.normal(base_volume, base_volume*0.1, len(dates))
        
        # Thêm yếu tố mùa vụ
        if seasonality == "Theo tuần":
            # Tăng vào đầu tuần, giảm vào cuối tuần
            weekly_pattern = np.array([1.2, 1.1, 1.0, 0.9, 0.8, 0.7, 0.7])
            seasonal_effect = np.tile(weekly_pattern, len(dates)//7 + 1)[:len(dates)]
            volumes = base_volumes * seasonal_effect
        elif seasonality == "Theo tháng":
            # Tăng vào đầu tháng, giảm vào cuối tháng
            day_of_month = dates.day
            monthly_effect = 1.2 - (day_of_month - 1) * 0.02
            volumes = base_volumes * monthly_effect
        else:
            volumes = base_volumes

        # Tạo DataFrame lịch sử
        df_history = pd.DataFrame({
            'Ngày': dates,
            'Thể tích': volumes
        })

        # Tính toán dự báo
        # 1. Moving Average (7 ngày)
        df_history['MA7'] = df_history['Thể tích'].rolling(window=7, center=True).mean()
        
        # 2. Exponential Smoothing
        alpha = 0.2  # Hệ số làm mượt
        df_history['EMA'] = df_history['Thể tích'].ewm(alpha=alpha, adjust=False).mean()
        
        # 3. Linear Trend
        X = np.arange(len(df_history)).reshape(-1, 1)
        y = df_history['Thể tích'].values
        model = LinearRegression()
        model.fit(X, y)
        df_history['Trend'] = model.predict(X)
        
        # Tạo dự báo cho tương lai
        future_dates = pd.date_range(
            start=dates[-1] + timedelta(days=1), 
            periods=forecast_days, 
            freq='D'
        )
        
        future_X = np.arange(len(df_history), len(df_history) + len(future_dates)).reshape(-1, 1)
        future_trend = model.predict(future_X)
        
        # Tạo DataFrame dự báo
        df_forecast = pd.DataFrame({
            'Ngày': future_dates,
            'Dự báo': future_trend
        })
        
        # Vẽ biểu đồ
        fig = go.Figure()
        
        # Dữ liệu lịch sử
        fig.add_trace(go.Scatter(
            x=df_history['Ngày'],
            y=df_history['Thể tích'],
            mode='markers+lines',
            name='Dữ liệu thực tế',
            line=dict(color='rgb(0, 102, 204)')
        ))
        
        # Moving Average
        fig.add_trace(go.Scatter(
            x=df_history['Ngày'],
            y=df_history['MA7'],
            mode='lines',
            name='Moving Average (7 ngày)',
            line=dict(color='rgb(255, 102, 102)')
        ))
        
        # Exponential Smoothing
        fig.add_trace(go.Scatter(
            x=df_history['Ngày'],
            y=df_history['EMA'],
            mode='lines',
            name='Exp. Smoothing',
            line=dict(color='rgb(51, 204, 51)')
        ))
        
        # Dự báo
        fig.add_trace(go.Scatter(
            x=df_forecast['Ngày'],
            y=df_forecast['Dự báo'],
            mode='lines',
            name='Dự báo',
            line=dict(color='rgb(255, 153, 51)', dash='dash')
        ))
        
        fig.update_layout(
            title='Dự báo nhu cầu vận chuyển',
            xaxis_title='Ngày',
            yaxis_title='Thể tích (m³)',
            hovermode='x unified',
            showlegend=True
        )
        
        st.plotly_chart(fig)
        
        # Hiển thị thống kê
        st.subheader("Thống kê dự báo")
        
        stats_col1, stats_col2, stats_col3 = st.columns(3)
        
        with stats_col1:
            st.metric(
                "Thể tích trung bình",
                f"{df_history['Thể tích'].mean():.1f} m³",
                f"{df_history['Thể tích'].std():.1f} m³"
            )
            
        with stats_col2:
            st.metric(
                "Dự báo trung bình",
                f"{df_forecast['Dự báo'].mean():.1f} m³",
                f"{df_forecast['Dự báo'].std():.1f} m³"
            )
            
        with stats_col3:
            trend_change = (df_forecast['Dự báo'].iloc[-1] - df_forecast['Dự báo'].iloc[0]) / df_forecast['Dự báo'].iloc[0] * 100
            st.metric(
                "Xu hướng",
                f"{trend_change:+.1f}%",
                "Tăng" if trend_change > 0 else "Giảm"
            )
        
        # Hiển thị bảng dự báo
        st.subheader("Bảng dự báo chi tiết")
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
