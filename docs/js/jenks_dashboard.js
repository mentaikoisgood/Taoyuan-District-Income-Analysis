// 3級Jenks分級分析儀表板 JavaScript
// 處理圖表渲染、交互功能和數據展示

let jenksChart = null;
let radarChart = null;

// 初始化儀表板
document.addEventListener('DOMContentLoaded', function() {
    console.log('🚀 初始化3級Jenks分級儀表板...');
    
    if (typeof DASHBOARD_DATA !== 'undefined') {
        initializeDashboard();
    } else {
        console.error('❌ 無法載入儀表板數據');
    }
});

function initializeDashboard() {
    // 更新統計卡片
    updateMetricCards();
    
    // 初始化圖表
    initializeJenksChart();
    initializeRadarChart();
    
    // 填充數據表格
    populateDataTable();
    
    // 設置區域選擇器
    setupDistrictSelector();
    
    console.log('✅ 儀表板初始化完成');
}

function updateMetricCards() {
    const stats = DASHBOARD_DATA.level_statistics;
    
    // 更新高潛力卡片
    document.getElementById('highCount').textContent = stats.high_potential.count;
    document.getElementById('highDistricts').textContent = stats.high_potential.districts.join('、');
    
    // 更新中潛力卡片
    document.getElementById('mediumCount').textContent = stats.medium_potential.count;
    document.getElementById('mediumDistricts').textContent = stats.medium_potential.districts.join('、');
    
    // 更新低潛力卡片
    document.getElementById('lowCount').textContent = stats.low_potential.count;
    document.getElementById('lowDistricts').textContent = stats.low_potential.districts.join('、');
}

function initializeJenksChart() {
    const ctx = document.getElementById('jenksChart').getContext('2d');
    const scatterData = DASHBOARD_DATA.scatter_data;
    
    // 按級別分組數據
    const groupedData = {
        '高潛力': [],
        '中潛力': [],
        '低潛力': []
    };
    
    scatterData.forEach(point => {
        groupedData[point.level].push({
            x: point.x,
            y: point.y,
            label: point.name
        });
    });
    
    const datasets = [
        {
            label: '高潛力',
            data: groupedData['高潛力'],
            backgroundColor: '#2E8B57',
            borderColor: '#2E8B57',
            pointRadius: 8,
            pointHoverRadius: 10
        },
        {
            label: '中潛力',
            data: groupedData['中潛力'],
            backgroundColor: '#FFA500',
            borderColor: '#FFA500',
            pointRadius: 8,
            pointHoverRadius: 10
        },
        {
            label: '低潛力',
            data: groupedData['低潛力'],
            backgroundColor: '#DC143C',
            borderColor: '#DC143C',
            pointRadius: 8,
            pointHoverRadius: 10
        }
    ];
    
    jenksChart = new Chart(ctx, {
        type: 'scatter',
        data: {
            datasets: datasets
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: {
                    title: {
                        display: true,
                        text: '排名'
                    },
                    min: 0,
                    max: 14
                },
                y: {
                    title: {
                        display: true,
                        text: '綜合分數'
                    },
                    min: 0,
                    max: 100
                }
            },
            plugins: {
                title: {
                    display: true,
                    text: '3級Jenks分級散點圖'
                },
                legend: {
                    display: true,
                    position: 'top'
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            const point = context.raw;
                            return `${point.label}: 排名${point.x}, 分數${point.y.toFixed(1)}`;
                        }
                    }
                }
            },
            animation: {
                duration: 1000
            }
        }
    });
}

function initializeRadarChart() {
    const ctx = document.getElementById('radarChart').getContext('2d');
    
    // 初始雷達圖（空數據）
    radarChart = new Chart(ctx, {
        type: 'radar',
        data: {
            labels: ['工作年齡人口比例', '商業集中度指數', '家戶中位數所得', '第三產業比例', '醫療指數'],
            datasets: []
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                r: {
                    beginAtZero: true,
                    max: 100,
                    ticks: {
                        stepSize: 20
                    }
                }
            },
            plugins: {
                title: {
                    display: true,
                    text: '請選擇行政區查看特徵分析'
                },
                legend: {
                    display: false
                }
            }
        }
    });
}

function setupDistrictSelector() {
    const selector = document.getElementById('districtSelect');
    
    // 清空現有選項
    selector.innerHTML = '<option value="">選擇行政區</option>';
    
    // 添加所有行政區選項
    DASHBOARD_DATA.districts.forEach(district => {
        const option = document.createElement('option');
        option.value = district.name;
        option.textContent = `${district.name} (${district.level}, ${district.score.toFixed(1)}分)`;
        selector.appendChild(option);
    });
    
    // 添加事件監聽器
    selector.addEventListener('change', function() {
        if (this.value) {
            updateRadarChart(this.value);
        } else {
            clearRadarChart();
        }
    });
}

function updateRadarChart(districtName) {
    const radarData = DASHBOARD_DATA.radar_data[districtName];
    
    if (!radarData) return;
    
    const colorMap = {
        '高潛力': 'rgba(46, 139, 87, 0.6)',
        '中潛力': 'rgba(255, 165, 0, 0.6)',
        '低潛力': 'rgba(220, 20, 60, 0.6)'
    };
    
    const borderColorMap = {
        '高潛力': 'rgba(46, 139, 87, 1)',
        '中潛力': 'rgba(255, 165, 0, 1)',
        '低潛力': 'rgba(220, 20, 60, 1)'
    };
    
    radarChart.data.datasets = [{
        label: districtName,
        data: radarData.values,
        backgroundColor: colorMap[radarData.level],
        borderColor: borderColorMap[radarData.level],
        borderWidth: 2,
        pointBackgroundColor: borderColorMap[radarData.level],
        pointBorderColor: '#fff',
        pointHoverBackgroundColor: '#fff',
        pointHoverBorderColor: borderColorMap[radarData.level]
    }];
    
    radarChart.options.plugins.title.text = `${districtName} - ${radarData.level} (${radarData.score.toFixed(1)}分)`;
    radarChart.update();
}

function clearRadarChart() {
    radarChart.data.datasets = [];
    radarChart.options.plugins.title.text = '請選擇行政區查看特徵分析';
    radarChart.update();
}

function populateDataTable() {
    const tableBody = document.getElementById('tableBody');
    const districts = DASHBOARD_DATA.districts;
    
    // 清空現有內容
    tableBody.innerHTML = '';
    
    // 按排名排序
    const sortedDistricts = [...districts].sort((a, b) => a.rank - b.rank);
    
    sortedDistricts.forEach(district => {
        const row = document.createElement('tr');
        
        // 設置行級別樣式
        const levelClass = {
            '高潛力': 'high-potential-row',
            '中潛力': 'medium-potential-row',
            '低潛力': 'low-potential-row'
        }[district.level];
        
        if (levelClass) {
            row.classList.add(levelClass);
        }
        
        row.innerHTML = `
            <td>${district.rank}</td>
            <td><strong>${district.name}</strong></td>
            <td>
                <span class="level-badge ${district.level_en}">
                    ${district.level}
                </span>
            </td>
            <td>${district.score.toFixed(1)}</td>
            <td>${district.features['工作年齡人口比例']?.toFixed(3) || 'N/A'}</td>
            <td>${district.features['家戶中位數所得']?.toFixed(0) || 'N/A'}</td>
            <td>${district.features['第三產業比例']?.toFixed(3) || 'N/A'}</td>
            <td>${district.features['醫療指數']?.toFixed(3) || 'N/A'}</td>
            <td>${district.features['商業集中度指數']?.toFixed(3) || 'N/A'}</td>
        `;
        
        // 添加點擊事件
        row.addEventListener('click', function() {
            document.getElementById('districtSelect').value = district.name;
            updateRadarChart(district.name);
            
            // 滾動到雷達圖
            document.getElementById('radarChart').scrollIntoView({ 
                behavior: 'smooth', 
                block: 'center' 
            });
        });
        
        tableBody.appendChild(row);
    });
}

// 工具函數：格式化數字
function formatNumber(num, decimals = 1) {
    if (num === null || num === undefined) return 'N/A';
    return Number(num).toFixed(decimals);
}

// 工具函數：獲取級別顏色
function getLevelColor(level) {
    const colors = {
        '高潛力': '#2E8B57',
        '中潛力': '#FFA500',
        '低潛力': '#DC143C'
    };
    return colors[level] || '#666';
}

// 響應式處理
window.addEventListener('resize', function() {
    if (jenksChart) jenksChart.resize();
    if (radarChart) radarChart.resize();
});

// 導出函數供外部使用
window.JenksDashboard = {
    updateRadarChart,
    clearRadarChart,
    formatNumber,
    getLevelColor
}; 