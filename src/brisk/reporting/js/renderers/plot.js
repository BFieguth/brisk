class PlotRenderer {
    constructor(plotData) {
        this.plotData = plotData;
    }

    render() {
        const template = document.getElementById('plot-template').content.cloneNode(true);
        this.renderPlotContent(template);
        this.renderDescription(template);
        return template;
    }

    renderPlotContent(template) {
        const plotContent = template.querySelector('.plot-content');
        
        // Create a div to hold the SVG content
        const plotDiv = document.createElement('div');
        plotDiv.className = 'plot-image';
        
        // Handle different SVG content formats
        if (this.plotData.image.startsWith('<svg') || this.plotData.image.startsWith('<?xml')) {
            // Direct SVG content (preferred method)
            plotDiv.innerHTML = this.plotData.image;
        } else if (this.plotData.image.startsWith('data:image/svg+xml;base64,')) {
            // Try Base64 decoding (fallback, but may fail)
            try {
                const base64Data = this.plotData.image.replace('data:image/svg+xml;base64,', '');
                const decodedSvg = atob(base64Data);
                plotDiv.innerHTML = decodedSvg;
            } catch (error) {
                console.error('Failed to decode Base64 SVG:', error);
                plotDiv.innerHTML = '<p>Error loading plot: Invalid Base64 encoding</p>';
            }
        } else {
            // Fallback for other formats
            const img = document.createElement('img');
            img.src = this.plotData.image;
            img.style.maxWidth = '100%';
            img.style.maxHeight = '100%';
            plotDiv.appendChild(img);
        }
        
        plotContent.appendChild(plotDiv);
    }

    renderDescription(template) {
        const descriptionElement = template.querySelector('.plot-description');
        if (this.plotData.description) {
            descriptionElement.textContent = this.plotData.description;
            descriptionElement.style.display = 'block';
        } else {
            descriptionElement.style.display = 'none';
        }
    }
}