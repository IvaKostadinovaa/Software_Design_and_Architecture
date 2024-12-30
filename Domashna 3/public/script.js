let homeTablePagination = {
    rowsPerPage: 10,
    currentPage: 1,
};

let analysisTablePagination = {
    rowsPerPage: 10,
    currentPage: 1,
};

function renderHomeTable(page = 1) {
    fetch('/api/stock-data')
        .then(response => response.json())
        .then(companies => {
            const totalCompanies = companies.length;
            const start = (page - 1) * homeTablePagination.rowsPerPage;
            const end = Math.min(start + homeTablePagination.rowsPerPage, totalCompanies);
            const paginatedCompanies = companies.slice(start, end);

            const tableBody = document.querySelector('#companyTable tbody');
            tableBody.innerHTML = ''; // Clear previous table rows

            paginatedCompanies.forEach(company => {
                const row = document.createElement('tr');
                row.innerHTML = `<td><a href="#" onclick="showCompanyDetails('${company.Код_на_издавач}')">${company.Код_на_издавач}</a></td>`;
                tableBody.appendChild(row);
            });

            renderPagination(
                totalCompanies,
                page,
                renderHomeTable,
                'pagination',
                homeTablePagination.rowsPerPage
            );
        })
        .catch(error => console.error('Error fetching companies:', error));
}

function renderAnalysisTable(page = 1) {
    fetch('/api/stock-data')
        .then(response => response.json())
        .then(data => {
            const totalItems = data.length;
            const start = (page - 1) * analysisTablePagination.rowsPerPage;
            const end = Math.min(start + analysisTablePagination.rowsPerPage, totalItems);
            const paginatedData = data.slice(start, end);

            const tableBody = document.querySelector('#analysisTable tbody');
            tableBody.innerHTML = '';

            paginatedData.forEach(entry => {
                const row = document.createElement('tr');
                row.innerHTML = `
                    <td>${entry.Код_на_издавач}</td>
                    <td>${entry.Датум}</td>
                    <td>${entry.Цена_на_последна_трансакција}</td>
                    <td>${entry.Макс}</td>
                    <td>${entry.Мин}</td>
                    <td>${entry.Просечна_цена}</td>
                    <td>${entry.Промет_во_БЕСТ_во_денари}</td>
                    <td>${entry.Купен_промет_во_денари}</td>
                    <td>${entry.Количина}</td>
                    <td>${entry.Промет_во_БЕСТ_во_денари_друга}</td>
                `;
                tableBody.appendChild(row);
            });

            renderPagination(
                totalItems,
                page,
                renderAnalysisTable,
                'analysisPagination',
                analysisTablePagination.rowsPerPage
            );
        })
        .catch(error => console.error('Error fetching analysis data:', error));
}

function renderPagination(totalItems, currentPage, onPageChange, paginationId, rowsPerPage) {
    const totalPages = Math.ceil(totalItems / rowsPerPage);
    const paginationDiv = document.getElementById(paginationId);
    paginationDiv.innerHTML = '';

    if (totalPages <= 1) return;


    const prevButton = document.createElement('button');
    prevButton.textContent = '← Previous';
    prevButton.disabled = currentPage === 1;
    prevButton.onclick = () => onPageChange(currentPage - 1);
    paginationDiv.appendChild(prevButton);


    const pageInfo = document.createElement('span');
    pageInfo.textContent = `Page ${currentPage} of ${totalPages}`;
    paginationDiv.appendChild(pageInfo);


    const nextButton = document.createElement('button');
    nextButton.textContent = 'Next →';
    nextButton.disabled = currentPage === totalPages;
    nextButton.onclick = () => onPageChange(currentPage + 1);
    paginationDiv.appendChild(nextButton);
}


function showCompanyDetails(companyCode) {
    fetch(`/api/stock-data/${companyCode}`)
        .then(response => response.json())
        .then(data => {
            showPage('details');
            initializeChart(data, `Stock Prices for ${companyCode}`);
        })
        .catch(error => console.error('Error fetching company details:', error));
}


function initializeChart(data, title) {
    const ctx = document.getElementById('companyChart').getContext('2d');
    if (window.stockChart) window.stockChart.destroy();

    const labels = data.map(entry => entry.Датум);
    const prices = data.map(entry => parseFloat(entry.Цена_на_последна_трансакција));

    window.stockChart = new Chart(ctx, {
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


function showPage(pageId) {
    document.querySelectorAll('.page').forEach(page => page.classList.add('hidden'));
    document.getElementById(pageId).classList.remove('hidden');

    if (pageId === 'home') renderHomeTable(homeTablePagination.currentPage);
    if (pageId === 'analysis') renderAnalysisTable(analysisTablePagination.currentPage);
}


document.addEventListener('DOMContentLoaded', () => showPage('home'));