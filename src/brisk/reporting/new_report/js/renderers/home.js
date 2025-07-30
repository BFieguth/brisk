class HomeRenderer {
    constructor(experiment_groups, selectedTable) {
        this.experiment_groups = experiment_groups;
        this.selectedTable = selectedTable;
    }

    render() {
        const template = document.getElementById('home-template').content.cloneNode(true);
        this.renderTables(template);
        this.renderExperimentGroupCards(template);
        return template;
    }

    renderTables(template) {
        const container = template.querySelector('.tables-container');
        const tableRenderer = new TableRenderer(this.selectedTable);
        const tableElement = tableRenderer.render();
        container.appendChild(tableElement);
    }

    renderExperimentGroupCards(template) {
        const container = template.querySelector('#cards-track');

        if (!this.experiment_groups || this.experiment_groups.length === 0) {
            return;
        }

        this.experiment_groups.forEach((experiment_group, index) => {
            const cardRenderer = new ExperimentGroupCardRenderer(
                experiment_group, index
            );
            const cardElement = cardRenderer.render();
            container.appendChild(cardElement);
        });
    }
}