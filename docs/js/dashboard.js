// 全局變數
let clusterChart;
let radarChart;

// 頁面載入完成後初始化
document.addEventListener('DOMContentLoaded', function() {
    initializeCharts();
    populateDataTable();
    initializeEventListeners();
});

// 初始化所有圖表
function initializeCharts() {
    createClusterChart();
    createRadarChart();
}

// 顏色轉換函數
function hexToRgba(hex, alpha) {
    const r = parseInt(hex.slice(1, 3), 16);
    const g = parseInt(hex.slice(3, 5), 16);
    const b = parseInt(hex.slice(5, 7), 16);
    return `rgba(${r}, ${g}, ${b}, ${alpha})`;
}

// 創建聚類散點圖
function createClusterChart() {
    const ctx = document.getElementById('clusterChart').getContext('2d');
    
    // 準備數據
    const datasets = [];
    const clusterStats = getClusterStats();
    
    Object.keys(clusterStats).forEach(level => {
        const color = clusterStats[level].color;
        const districts = clusterStats[level].districts;
        const data = districts.map(district => {
            const districtInfo = getDistrictData(district);
            return {
                x: districtInfo.tsne_x,
                y: districtInfo.tsne_y,
                label: district
            };
        });
        
        datasets.push({
            label: level,
            data: data,
            backgroundColor: color,
            borderColor: color,
            pointRadius: 8,
            pointHoverRadius: 12
        });
    });
    
    clusterChart = new Chart(ctx, {
        type: 'scatter',
        data: { datasets: datasets },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'top',
                    labels: {
                        usePointStyle: true,
                        padding: 20
                    }
                },
                tooltip: {
                    callbacks: {
                        title: function(context) {
                            return context[0].raw.label;
                        },
                        label: function(context) {
                            const district = context.raw.label;
                            const data = getDistrictData(district);
                            return [
                                `潛力等級: ${data.潛力等級}`,
                                `潛力分數: ${data.潛力分數.toFixed(3)}`,
                                `家庭收入: ${data.家庭收入中位數.toLocaleString()}元`,
                                `分群機率: ${(data.分群機率 * 100).toFixed(1)}%`
                            ];
                        }
                    }
                }
            },
            scales: {
                x: {
                    title: {
                        display: true,
                        text: 't-SNE 維度 1'
                    }
                },
                y: {
                    title: {
                        display: true,
                        text: 't-SNE 維度 2'
                    }
                }
            }
        }
    });
}

// 創建雷達圖
function createRadarChart() {
    const ctx = document.getElementById('radarChart').getContext('2d');
    
    radarChart = new Chart(ctx, {
        type: 'radar',
        data: {
            labels: ['勞動年齡比例', '商業集中度', '家庭收入中位數', '第三產業比例', '醫療指數'],
            datasets: [{
                label: '選擇行政區',
                data: [0, 0, 0, 0, 0],
                backgroundColor: 'rgba(52, 152, 219, 0.2)',
                borderColor: 'rgba(52, 152, 219, 1)',
                borderWidth: 2,
                pointBackgroundColor: 'rgba(52, 152, 219, 1)',
                pointBorderColor: '#fff',
                pointHoverBackgroundColor: '#fff',
                pointHoverBorderColor: 'rgba(52, 152, 219, 1)'
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'top'
                }
            },
            scales: {
                r: {
                    beginAtZero: true,
                    max: 1,
                    ticks: {
                        stepSize: 0.2
                    }
                }
            }
        }
    });
}

// 更新雷達圖數據
function updateRadarChart(districtName) {
    if (!districtName) {
        radarChart.data.datasets[0].data = [0, 0, 0, 0, 0];
        radarChart.data.datasets[0].label = '選擇行政區';
        radarChart.update();
        return;
    }
    
    const data = getDistrictData(districtName);
    const features = ['勞動年齡比例', '商業集中度', '家庭收入中位數', '第三產業比例', '醫療指數'];
    
    // 標準化數據到0-1範圍
    const normalizedData = features.map(feature => normalizeFeature(data[feature], feature));
    
    // 根據潛力等級設定顏色
    let color = '#3498db';
    if (data.潛力等級 === '高潛力') color = '#e74c3c';
    else if (data.潛力等級 === '中潛力') color = '#f39c12';
    
    radarChart.data.datasets[0].data = normalizedData;
    radarChart.data.datasets[0].label = `${districtName} (${data.潛力等級}, 分數: ${data.潛力分數.toFixed(3)})`;
    radarChart.data.datasets[0].backgroundColor = hexToRgba(color, 0.2);
    radarChart.data.datasets[0].borderColor = color;
    radarChart.data.datasets[0].pointBackgroundColor = color;
    radarChart.data.datasets[0].pointHoverBorderColor = color;
    
    radarChart.update();
}

// 填充數據表格
function populateDataTable() {
    const tbody = document.getElementById('tableBody');
    const districts = getAllDistricts();
    
    tbody.innerHTML = '';
    
    districts.forEach(district => {
        const data = getDistrictData(district);
        const row = document.createElement('tr');
        
        // 潛力等級標籤
        let levelClass = 'low-potential-label';
        if (data.潛力等級 === '高潛力') levelClass = 'high-potential-label';
        else if (data.潛力等級 === '中潛力') levelClass = 'medium-potential-label';
        
        row.innerHTML = `
            <td>${district}</td>
            <td><span class="${levelClass}">${data.潛力等級}</span></td>
            <td>${data.集群編號}</td>
            <td>${data.潛力分數.toFixed(3)}</td>
            <td>${data.家庭收入中位數.toLocaleString()}</td>
            <td>${data.勞動年齡比例.toFixed(1)}%</td>
            <td>${data.第三產業比例.toFixed(1)}%</td>
            <td>${data.醫療指數.toFixed(3)}</td>
            <td>${(data.分群機率 * 100).toFixed(1)}%</td>
        `;
        
        tbody.appendChild(row);
    });
}

// 初始化事件監聽器
function initializeEventListeners() {
    const districtSelect = document.getElementById('districtSelect');
    
    districtSelect.addEventListener('change', function() {
        const selectedDistrict = this.value;
        updateRadarChart(selectedDistrict);
        
        // 高亮表格中對應的行
        highlightTableRow(selectedDistrict);
    });
}

// 高亮表格行
function highlightTableRow(districtName) {
    const tbody = document.getElementById('tableBody');
    const rows = tbody.getElementsByTagName('tr');
    
    // 移除所有高亮
    Array.from(rows).forEach(row => {
        row.style.backgroundColor = '';
    });
    
    if (!districtName) return;
    
    // 高亮選中的行
    Array.from(rows).forEach(row => {
        if (row.cells[0].textContent === districtName) {
            row.style.backgroundColor = '#e3f2fd';
        }
    });
}

// 格式化數字
function formatNumber(num, decimals = 0) {
    return num.toLocaleString('zh-TW', {
        minimumFractionDigits: decimals,
        maximumFractionDigits: decimals
    });
}

// 獲取特徵描述
function getFeatureDescription(feature, value) {
    const descriptions = {
        '勞動年齡比例': value >= 72.5 ? '高' : value >= 71.5 ? '中' : '低',
        '商業集中度': value >= 7.95 ? '高' : value >= 7.85 ? '中' : '低',
        '家庭收入中位數': value >= 480000 ? '高' : value >= 440000 ? '中' : '低',
        '第三產業比例': value >= 85 ? '高' : value >= 80 ? '中' : '低',
        '醫療指數': value >= 0.2 ? '高' : value >= 0.1 ? '中' : '低'
    };
    
    return descriptions[feature] || '未知';
} 