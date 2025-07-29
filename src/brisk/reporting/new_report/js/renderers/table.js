class TableRenderer {
    constructor(tableData) {
        this.tableData = tableData;
    }

    render() {
        const template = document.getElementById('table-template').content.cloneNode(true);
        this.renderHeaders(template);
        this.renderRows(template);
        this.renderDescription(template);
        return template;
    }

    renderHeaders(template) {
        const headerRow = template.querySelector('.table-header-row');
        this.tableData.columns.forEach(column => {
            const th = document.createElement('th');
            th.textContent = column;
            headerRow.appendChild(th);
        });
    }

    renderRows(template) {
        const tbody = template.querySelector('.table-body');
        this.tableData.rows.forEach(rowData => {
            const row = document.createElement('tr');

            rowData.forEach(cellData => {
                const td = document.createElement('td');
                td.textContent = cellData;
                row.appendChild(td);
            });
            tbody.appendChild(row);
        });
    }

    renderDescription(template) {
        const descriptionElement = template.querySelector('.table-description');
        if (this.tableData.description) {
            descriptionElement.textContent = this.tableData.description;
        }
    }
}

