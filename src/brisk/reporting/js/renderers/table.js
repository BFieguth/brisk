class TableRenderer {
    constructor(tableData) {
        this.tableData = tableData;
    }

    render() {
        const template = document.getElementById('table-template').content.cloneNode(true);
        this.renderHeaders(template);
        this.renderRows(template);
        this.renderDescription(template);
        this.setupCopyButton(template);
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

    setupCopyButton(template) {
        const copyButton = template.querySelector('.container');
        const checkbox = template.querySelector('.copy-checkbox');
        
        if (copyButton && checkbox) {
            copyButton.addEventListener('click', (e) => {
                e.preventDefault();
                this.copyTableAsCSV(checkbox);
            });
        }
    }

    convertTableToCSV() {
        let csvContent = this.tableData.columns.join(',') + '\n';
        
        this.tableData.rows.forEach(row => {
            const escapedRow = row.map(cell => {
                const cellStr = String(cell);
                if (cellStr.includes(',') || cellStr.includes('"') || cellStr.includes('\n')) {
                    return '"' + cellStr.replace(/"/g, '""') + '"';
                }
                return cellStr;
            });
            csvContent += escapedRow.join(',') + '\n';
        });
        
        return csvContent;
    }

    async copyTableAsCSV(checkbox) {
        try {
            const csvData = this.convertTableToCSV();
            
            if (navigator.clipboard && navigator.clipboard.writeText) {
                await navigator.clipboard.writeText(csvData);
                checkbox.checked = true;
                setTimeout(() => {
                    checkbox.checked = false;
                }, 2000);
            } else {
                console.warn('Clipboard API not supported');
                checkbox.checked = false;
            }
        } catch (error) {
            console.error('Failed to copy table data:', error);
            checkbox.checked = false;
        }
    }
}
