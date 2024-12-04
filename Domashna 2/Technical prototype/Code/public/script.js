let stockChart;
let analysisChart;
const rowsPerPage = 10; // Pagination rows per page

// Render the company table with pagination
function renderCompanyTable(page = 1) {
    fetch('/api/stock-data')
        .then(response => response.json())
        .then(companies => {
            const start = (page - 1) * rowsPerPage;
            const end = start + rowsPerPage;
            const paginatedCompanies = companies.slice(start, end);

            const tableBody = document.querySelector('#companyTable tbody');
            tableBody.innerHTML = '';

            paginatedCompanies.forEach(company => {
                const row = document.createElement('tr');
                row.innerHTML = `<td><a href="#" onclick="showCompanyDetails('${company.Код_на_издавач}')">${company.Код_на_издавач}</a></td>`;
                tableBody.appendChild(row);
            });

            renderPagination(companies.length, page);
        })
        .catch(error => console.error('Error fetching companies:', error));
}

// Render pagination buttons
function renderPagination(totalItems, currentPage) {
    const totalPages = Math.ceil(totalItems / rowsPerPage);
    const paginationDiv = document.getElementById('pagination');
    paginationDiv.innerHTML = '';

    const prevButton = document.createElement('button');
    prevButton.textContent = '←';
    prevButton.disabled = currentPage === 1;
    prevButton.onclick = () => renderCompanyTable(currentPage - 1);
    paginationDiv.appendChild(prevButton);

    const pageIndicator = document.createElement('span');
    pageIndicator.textContent = ` Page ${currentPage} of ${totalPages} `;
    paginationDiv.appendChild(pageIndicator);

    const nextButton = document.createElement('button');
    nextButton.textContent = '→';
    nextButton.disabled = currentPage === totalPages;
    nextButton.onclick = () => renderCompanyTable(currentPage + 1);
    paginationDiv.appendChild(nextButton);
}

// Show the specific company details page
function showCompanyDetails(companyName) {
    fetch(`/api/stock-data/${companyName}`)
        .then(response => response.json())
        .then(data => {
            showPage('details');
            initializeChart(data, `Stock Prices for ${companyName}`);
        })
        .catch(error => console.error('Error fetching company details:', error));
}

// Initialize a specific company stock chart
function initializeChart(data, title) {
    const ctx = document.getElementById('companyChart').getContext('2d');
    if (stockChart) stockChart.destroy();

    const labels = data.map(entry => entry.Датум);
    const prices = data.map(entry => entry.Цена_на_последна_трансакција);

    stockChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [{
                label: title,
                data: prices,
                borderColor: 'rgba(75, 192, 192, 1)',
                fill: false,
            }]
        },
        options: {
            responsive: true,
        }
    });
}

// Initialize analysis chart
function initializeAnalysisChart() {
    fetch('/api/stock-data')
        .then(response => response.json())
        .then(companies => {
            const ctx = document.getElementById('analysisChart').getContext('2d');
            if (analysisChart) analysisChart.destroy();

            const datasets = companies.map(company => ({
                label: company.Код_на_издавач,
                data: company.data.map(item => item.Цена_на_последна_трансакција),
                borderColor: getRandomColor(),
                fill: false,
            }));

            analysisChart = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: companies[0].data.map(item => item.Датум),
                    datasets: datasets,
                },
            });
        })
        .catch(error => console.error('Error initializing analysis chart:', error));
}

// Get random color for chart
function getRandomColor() {
    return `#${Math.floor(Math.random() * 16777215).toString(16)}`;
}

// Show the requested page
function showPage(pageId) {
    document.querySelectorAll('.page').forEach(page => page.classList.add('hidden'));
    document.getElementById(pageId).classList.remove('hidden');

    if (pageId === 'home') renderCompanyTable();
    if (pageId === 'analysis') initializeAnalysisChart();
}

// Initialize on page load
document.addEventListener('DOMContentLoaded', () => showPage('home'));
