class HomeRenderer {
    constructor(data) {
        this.data = data;
    }

    render() {
        const template = document.getElementById('home-template').content.cloneNode(true);
        this.renderTables(template);
        this.renderExperimentGroupCards(template);
        return template;
    }

    renderTables(template) {
        const container = template.querySelector('.tables-container');

        if (!this.data.tables || this.data.tables.length === 0) {
            return;
        }

        this.data.tables.forEach(tableData => {
            const tableRenderer = new TableRenderer(tableData);
            const tableElement = tableRenderer.render();
            container.appendChild(tableElement);
        });
    }

    renderExperimentGroupCards(template) {
        const container = template.querySelector('#cards-track');

        if (!this.data.experiment_group_cards || Object.keys(this.data.experiment_group_cards).length === 0) {
            return;
        }

        Object.entries(this.data.experiment_group_cards).forEach(([groupName, cardData], index) => {
            const cardRenderer = new ExperimentGroupCardRenderer(cardData, index);
            const cardElement = cardRenderer.render();
            container.appendChild(cardElement);
        });
    }
}