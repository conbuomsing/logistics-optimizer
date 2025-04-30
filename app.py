import streamlit as st
import pandas as pd
import pulp
import plotly.express as px

st.set_page_config(page_title="Tối ưu Logistics", layout="wide")

st.title("Ứng dụng Tối ưu Logistics Dệt May")

# Định nghĩa thông tin phương tiện
vehicles = {
    'Xe 1.5 tấn': {'capacity': 1.5, 'cost_nb': 2500000, 'cost_hp': 3500000},
    'Xe 1.9 tấn': {'capacity': 1.9, 'cost_nb': 2800000, 'cost_hp': 3800000},
    'Xe 3.5 tấn': {'capacity': 3.5, 'cost_nb': 3500000, 'cost_hp': 4500000},
    'Container 20ft': {'capacity': 28, 'cost_nb': 15000000, 'cost_hp': 18000000},
    'Container 40ft': {'capacity': 65, 'cost_nb': 25000000, 'cost_hp': 28000000}
}

# Input form
with st.form("logistics_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        cargo_weight = st.number_input("Tổng khối lượng hàng (tấn)", min_value=0.1)
        route = st.selectbox("Tuyến đường", ["Nội Bài", "Hải Phòng"])
    
    submitted = st.form_submit_button("Tối ưu hóa")

if submitted:
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
    
    # Ràng buộc về khối lượng
    prob += pulp.lpSum(vehicles[v]['capacity'] * vehicle_vars[v] for v in vehicles.keys()) >= cargo_weight
    
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
            'Tổng tải trọng': capacity
        })
        
        total_cost += cost
        total_capacity += capacity
    
    # Hiển thị bảng kết quả
    df_results = pd.DataFrame(results)
    st.subheader("Kết quả tối ưu")
    st.dataframe(df_results)
    
    # Hiển thị tổng hợp
    col1, col2, col3 = st.columns(3)
    col1.metric("Tổng chi phí", f"{total_cost:,.0f} VNĐ")
    col2.metric("Tổng tải trọng", f"{total_capacity:.1f} tấn")
    col3.metric("Tải trọng yêu cầu", f"{cargo_weight:.1f} tấn")
    
    # Biểu đồ phân bổ
    fig = px.bar(df_results, 
                 x='Phương tiện', 
                 y='Số lượng',
                 title='Phân bổ phương tiện')
    st.plotly_chart(fig)
