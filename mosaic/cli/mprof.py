
import click
import dash
import dash_cytoscape as cyto
from dash.dependencies import Input, Output, State

from ..profile import global_profiler
from ..types import Struct


graph_stylesheet = [
        # Group selectors
        {
            'selector': 'node',
            'style': {
                'content': 'data(label)',
            }
        },
        {
            'selector': '.node',
            'style': {
                'content': 'data(label)',
                'font-size': '12px',
                'text-wrap': 'wrap',
                'text-halign': 'center',
                'text-valign': 'center',
                'line-height': '2px',
                'width': 'label',
                'height': 'label',
                'padding': '10px',
                'shape': 'round-rectangle',
                'background-color': 'mapData(t_elapsed_percent, 0, 100, #3496C3, #EA5B40)',
            }
        },
        {
            'selector': '.compound',
            'style': {
                'content': 'data(short_label)',
                'text-rotation': '-90deg',
                'font-size': '12px',
                'text-wrap': 'none',
                'text-halign': 'left',
                'text-valign': 'center',
                'padding': 'auto',
                'text-margin-x': '-10px',
                'text-margin-y': '-50%',
            }
        },
        {
            'selector': '.r-edge',
            'style': {
                'label': 'data(label)',
                'curve-style': 'bezier',
                'line-style': 'dashed',
                'target-arrow-shape': 'triangle',
                'arrow-scale': '1.2',
                'text-rotation': 'autorotate',
                'font-size': '10px',
                'text-background-color': 'white',
                'text-background-padding': '5px',
                'text-background-opacity': '1',
            }
        },
    ]

node_data_stylesheet = {
        'node-data': {
            'position': 'fixed',
            'width': '500px',
            'height': '300px',
            'padding-bottom': '15px',
            'top': '50px',
            'right': '50px',
            'overflow-x': 'hidden',
            'overflow-y': 'auto',
            'border': 'solid 1px #F3F2F1',
            'box-shadow': '-2px 2px 5px 0px rgba(233,233,233,0.74)',
            'background-color': 'white',

        },
        'node-title': {
            'font-size': '16px',
            'font-family': 'sans-serif',
            'height': '50px',
            'padding': '10px 15px',
            'background-color': '#F3F2F1',
            'margin-top': '0',
        },
        'row': {
            'display': 'block',
            'border-bottom': 'solid 1px white',
            'padding-left': '15px',
        },
        'label': {
            'display': 'inline-block',
            'width': '100px',
            'background-color': '#F3F2F1',
            'padding': '10px',
            'font-weight': 'bold',
            'font-size': '12px',
            'font-family': 'sans-serif'
        },
        'value': {
            'display': 'inline-block',
            'width': '320px',
            'padding': '10px',
            'font-size': '12px',
            'font-family': 'sans-serif'
        },
    }

node_fields = [
    ('id', 'ID'),
    ('runtime_id', 'Runtime'),
    ('parent', 'Parent node'),
    ('filename', 'Filename'),
    ('lineno', 'Line number'),
    ('t_elapsed', 'Time (s)'),
    ('t_elapsed_percent', 'Time (%)'),
]


def _generate_node():
    rows = []

    for field in node_fields:
        label = field[1]
        field = field[0]

        rows.append(
            dash.html.Div(
                [
                    dash.html.Span(label, id='node-%s-label' % field, style=node_data_stylesheet['label'], ),
                    dash.html.Span(id='node-%s' % field, style=node_data_stylesheet['value'], ),
                ],
                style=node_data_stylesheet['row'],
            ),
        )

    return rows


class NodeSet:

    def __init__(self):
        self.nodes = []
        self.edges = []
        self.r_calls = []
        self.parent = None

    def append_node(self, node):
        self.nodes.append(node)

    def append_edge(self, edge):
        self.edges.append(edge)


cyto.load_extra_layouts()

app = dash.Dash(__name__, title='Mosaic Profiling',)


@click.command()
@click.argument('filename', required=True, nargs=1)
@click.version_option()
def go(filename):
    plot_nodes = dict()
    global_profiler.set_local()
    global_profiler.load(filename=filename.rstrip('/'))

    total_time = global_profiler.profiler.t_elapsed
    profiles = global_profiler.profiler.profiles

    def _add_r_call(runtime_id, source, target, call_id, return_id, label, node):
        node.r_calls.append({
            'source': source,
            'target': target,
            'call_id': call_id,
            'return_id': return_id,
            'label': label,
        })

        if node.parent:
            source = node.parent
            _add_r_call(runtime_id,
                        source, target,
                        call_id, return_id,
                        label,
                        plot_nodes[runtime_id][node.parent])

    def _add_plot_node(runtime_id, key, node, graph):
        plot_nodes[runtime_id][key] = NodeSet()
        plot_nodes[runtime_id][key].parent = node['data'].get('parent', None)

        prev_key = None

        if 'target' in graph and graph['target'] != 'monitor':
            msg_id = key
            target_runtime_id = graph['target']
            call_id = key
            return_id = '%s.%s' % (target_runtime_id, msg_id)

            target_node = None
            for trace_id, trace in profiles[target_runtime_id].items():
                for node_id, prospect_node in trace.items():
                    if node_id == return_id:
                        target_node = prospect_node
                        break

                if target_node:
                    break

            if target_node is not None:
                t_elapsed = target_node['t_end'] - graph['t_start']

                source = key
                target = target_node['trace_id']
                label = node['data']['name'] + ' (%f s)' % t_elapsed

                _add_r_call(runtime_id,
                            source, target,
                            call_id, return_id,
                            label,
                            plot_nodes[runtime_id][key])

        for child_key, child_graph in graph.items():
            if not isinstance(child_graph, (dict, Struct)):
                continue

            try:
                node_name = child_graph.get('name', child_key)
            except KeyError:
                continue

            long_label = node_name
            short_label = node_name

            t_elapsed = -1
            t_elapsed_percent = -1

            if 't_elapsed' in child_graph:
                t_elapsed = child_graph['t_elapsed']
                t_elapsed_percent = child_graph['t_elapsed'] / total_time * 100

                long_label += ' \n(%f%%)' % t_elapsed_percent
                short_label += ' (%f%%)' % t_elapsed_percent

            child_node = {
                'data': {
                    'id': child_key,
                    'name': node_name,
                    'filename': child_graph.get('filename', ''),
                    'lineno': child_graph.get('lineno', -1),
                    'label': long_label,
                    'short_label': short_label,
                    'parent': key,
                    'runtime_id': runtime_id,
                    't_elapsed': t_elapsed,
                    't_elapsed_percent': t_elapsed_percent,
                    'expanded': False,
                },
                'classes': 'node',
            }
            plot_nodes[runtime_id][key].append_node(child_node)

            if prev_key is not None:
                edge = {
                    'data': {'source': prev_key, 'target': child_key}
                }
                plot_nodes[runtime_id][key].append_edge(edge)

            prev_key = child_key
            _add_plot_node(runtime_id, child_key, child_node, child_graph)

        return prev_key

    plot_elements = []

    for runtime_id in profiles.keys():
        if runtime_id == 'monitor':
            continue

        if not len(profiles[runtime_id].keys()):
            continue

        node = {
            'data': {
                'id': runtime_id,
                'name': runtime_id,
                'label': runtime_id,
                'short_label': runtime_id,
                'expanded': True
            },
            'classes': 'compound',
        }

        plot_nodes[runtime_id] = dict()
        _add_plot_node(runtime_id, runtime_id, node, profiles[runtime_id])

        for trace_index, trace_id in enumerate(profiles[runtime_id].keys()):
            for r_call in plot_nodes[runtime_id][trace_id].r_calls:
                r_edge = {
                    'data': {'source': r_call['source'], 'target': r_call['target'],
                             'label': r_call['label']},
                    'classes': 'r-edge',
                }
                plot_elements.append(r_edge)

        plot_nodes[runtime_id][runtime_id].nodes.insert(0, node)

        plot_elements += plot_nodes[runtime_id][runtime_id].nodes + plot_nodes[runtime_id][runtime_id].edges

    app.layout = dash.html.Div(
        [
            cyto.Cytoscape(
                id='cytoscape',
                elements=plot_elements,
                style={'width': '100%', 'height': '100vh'},
                layout={
                    'name': 'dagre',
                    'fit': False,
                    'nodeDimensionsIncludeLabels': True,
                    'rankDir': 'TB',
                    'animate': False,
                    'ranker': 'network-simplex',
                    'nodeSep': 50,
                    'edgeSep': 80,
                },
                stylesheet=graph_stylesheet,
                autounselectify=True,
            ),
            dash.html.Div(
                [
                    dash.html.H2(id='node-title', style=node_data_stylesheet['node-title'], ),
                ] + _generate_node(),
                id='node-data',
                style=node_data_stylesheet['node-data'],
            )
        ]
    )

    @app.callback(Output('cytoscape', 'elements'),
                  Input('cytoscape', 'tapNodeData'),
                  State('cytoscape', 'elements'))
    def generate_elements(node_data, elements):
        if not node_data:
            return plot_elements

        runtime_id = node_data['runtime_id']
        node_id = node_data['id']

        if node_id not in plot_nodes[runtime_id]:
            return plot_elements

        # Find sub-graph
        sub_nodes = plot_nodes[runtime_id][node_id].nodes
        sub_edges = plot_nodes[runtime_id][node_id].edges

        if not sub_nodes:
            return elements

        # Retrieve the currently selected element, and toggle expanded
        index = -1
        for index, element in enumerate(elements):
            if node_id == element.get('data').get('id'):
                element['data']['expanded'] = not element['data']['expanded']

                if not node_data.get('expanded'):
                    element['classes'] = 'compound'
                else:
                    element['classes'] = 'node'

                break

        # Check if node has been expanded
        if not node_data.get('expanded'):
            if sub_nodes:
                for sub_index, node in enumerate(sub_nodes):
                    elements.insert(index + sub_index + 1, node)

                # Find the first node in the subgraph
                sub_first_node = sub_nodes[0]

                # Find the previous edge in the current graph
                for element in elements:
                    if element.get('classes') != 'r-edge' and node_id == element['data'].get('target'):
                        element['data']['target'] = sub_first_node['data']['id']
                        break

                # Find the last node in the subgraph
                sub_last_node = sub_nodes[-1]

                # Find the next edge in the current graph
                for element in elements:
                    if element.get('classes') != 'r-edge' and node_id == element['data'].get('source'):
                        element['data']['source'] = sub_last_node['data']['id']
                        break

                # Find whether there are any r_calls in the sub_nodes
                for node in sub_nodes:
                    for r_call in plot_nodes[runtime_id][node['data']['id']].r_calls:
                        for element in elements:
                            if element.get('classes') == 'r-edge' and element['data'].get('target') == r_call['target']:
                                del element['data']['id']
                                element['data']['source'] = r_call['source']
                                break

            if sub_edges:
                elements.extend(sub_edges)

        else:
            if sub_nodes:
                for node in sub_nodes:
                    elements.remove(node)

                # Find the first node in the subgraph
                sub_first_node = sub_nodes[0]

                # Find the previous edge in the current graph
                for element in elements:
                    if element.get('classes') != 'r-edge' and sub_first_node['data']['id'] == element['data'].get('target'):
                        del element['data']['id']
                        element['data']['target'] = node_id
                        break

                # Find the last node in the subgraph
                sub_last_node = sub_nodes[-1]

                # Find the last edge in the current graph
                for element in elements:
                    if element.get('classes') != 'r-edge' and sub_last_node['data']['id'] == element['data'].get('source'):
                        del element['data']['id']
                        element['data']['source'] = node_id
                        break

                # Find whether there are any r_calls in the sub_nodes
                for node in sub_nodes:
                    for r_call in plot_nodes[runtime_id][node['data']['id']].r_calls:
                        for element in elements:
                            if element.get('classes') == 'r-edge' and element['data'].get('target') == r_call['target']:
                                del element['data']['id']
                                element['data']['source'] = node_id
                                break

            if sub_edges:
                for edge in sub_edges:
                    source = edge['data']['source']
                    target = edge['data']['target']

                    for element in elements:
                        if element['data'].get('source') == source \
                                and element['data'].get('target') == target:
                            elements.remove(element)

        return elements

    outputs = []
    for field in node_fields:
        field = field[0]

        outputs.append(Output('node-%s' % field, 'children'))

    @app.callback(Output('node-data', 'style'),
                  Output('node-title', 'children'),
                  *outputs,
                  Input('cytoscape', 'mouseoverNodeData'),
                  Input('node-data', 'style'))
    def display_node_data(node_data, style):
        title = 'NODE — '

        if not node_data:
            style['display'] = 'none'

            return (style, title) + (None,) * len(node_fields)

        title += node_data['name']
        style['display'] = 'block'

        return (style, title) + tuple([node_data[field[0]] if field[0] in node_data else None
                                       for field in node_fields])

    app.run_server(debug=False)


if __name__ == '__main__':
    go(auto_envvar_prefix='MOSAIC')
