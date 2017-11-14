
let cytoscape = require('cytoscape');
let dagre = require('cytoscape-dagre');


cytoscape.use(dagre);


class CytoscapeElement extends Buche.BucheElement {

    setupEnd() {
        this.cy = cytoscape(this.options);
    }

    template(children) {

        this.style.position = "relative";

        let options = {
            container: this,
            boxSelectionEnabled: true,
            autounselectify: false,
            elements: [],
            style: "",
            layout: {name: this.getAttribute('layout') || 'cose'}
        };

        for (let child of children) {
            let tag = child.tagName;
            let d = null;
            switch (tag) {
            case 'STYLE':
                options.style += child.textContent;
                break;
            case 'OPTIONS':
                d = JSON.parse(child.textContent);
                options = Object.assign(options, d);
                break;
            case 'ELEMENT':
                // console.log(child.textContent);
                d = JSON.parse(child.textContent);
                // console.log(d);
                options.elements.push(d);
                break;
            }
        }

        this.options = options;

        return [];
    }

    addItem(item) {
        this.cy.add(item);
    }

    css() {
        return {
            "graph-element": {
                position: "relative",
                display: "block",
                border: "1px solid green",
                width: "500px",
                height: "500px"
            }
        }
    }
}



module.exports = {
    isBuchePlugin: true,
    components: {
        'graph-element': CytoscapeElement
    }
}
