
import copy
import click
import dash
from matplotlib import cm
from dash.dependencies import Input, Output, State
from collections import OrderedDict

from ..file_manipulation import h5


class DisplayElement:

    def __init__(self, label_id, label):
        self.label_id = label_id
        self.label = label
        self.elements = []
        self.html = None
        self.sub_elements = OrderedDict()


app = dash.Dash(__name__, title='Mosaic Profiling',)


# EVENTS
#
# * Tessera
#   Remote:  init --> (listening --> running) --> collected
#   Proxy:   pending --> init --> listening --> collected
#
# * Task
#   Remote:  init --> pending --> ready --> running --> done --> collected
#   Proxy:   pending --> init --> queued --> (done --> result) --> collected


@click.command()
@click.argument('filename', required=True, nargs=1)
@click.version_option()
def go(filename):
    with h5.HDF5(filename=filename, mode='r') as file:
        profile = file.load()

    cmap = cm.get_cmap('plasma')

    # General parameters
    start_t = profile.start_t
    end_t = profile.end_t
    total_t = (end_t - start_t)*1.10
    total_width = 100

    # Style configuration
    stylesheet = {
        'content': {
            'display': 'flex',
            'width': '100%',
            'height': '100vh',
        },
        'sidebar': {
            'height': '100vh',
            'width': '300px',
            'borderRight': 'solid 1px gray',
            'flexGrow': '0',
            'flexShrink': '0',
        },
        'main': {
            'height': '100vh',
            'flexGrow': '1',
            'overflow': 'auto',
            'position': 'relative',
        },
        'inner-main': {
            'position': 'relative',
            'width': '%f%%' % total_width,
            'transform': 'translateZ(0)',
            'padding': '30px 0 0',
        },
        'ruler': {
            'position': 'fixed',
            'top': '0',
            'left': '0',
            'width': '100%',
            'height': '100%',
        },
        'divisor-label': {
            'fontSize': '12px',
            'fontStyle': 'normal',
            'display': 'inline-block',
            'padding': '10px',
            'color': '#676767',
        },
        'runtime': {
            'padding': '15px 0',
            'borderBottom': 'solid 1px #d4d4d4',
            'position': 'relative',
            'minHeight': '85px',
        },
        'runtime-label': {
            'width': '550px',
            'fontWeight': 'bold',
            'margin': '0 0 15px',
            'padding': '0 15px',
            'display': 'block',
            'cursor': 'pointer',
            'position': 'relative',
        },
        'runtime-row': {
            'position': 'relative',
            'height': '65px',
            'padding': '5px 15px',
            'display': 'none',
        },
        'task-row': {
            'backgroundColor': 'rgba(249, 249, 249, 0.6)',
        },
        'row-label': {
            'width': '550px',
            'display': 'block',
            'fontStyle': 'normal',
            'fontSize': '14px',
            'cursor': 'pointer',
            'position': 'relative',
        },
        'event': {
            'display': 'inline-block',
            'width': '2px',
            'minWidth': '3px',
            'height': '35px',
            'borderLeft': 'solid 0.5px white',
            'borderRight': 'solid 0.5px white',
            'position': 'absolute',
            'bottom': '5px',
            'fontSize': '12px',
            'padding': '5px 0px',
            'textIndent': '5px',
            'color': 'white',
            'cursor': 'default',
        },
        'event-label': {
            'fontSize': '12px',
            'color': 'white',
            'fontStyle': 'normal',
            'display': 'block',
            'overflow': 'hidden',
            'cursor': 'default',
            'padding': '0 2px 0 0',
        },
        'tooltip': {
            'display': 'none',
            'width': 'auto',
            'minWidth': '500px',
            'height': 'auto',
            'position': 'absolute',
            'left': '0',
            'top': '35px',
            'backgroundColor': 'white',
            'fontSize': '12px',
            'color': 'black',
            'padding': '15px',
            'boxShadow': '-2px 2px 5px 0px rgba(233,233,233,0.74)',
            'zIndex': '9999',
        },
        'tooltip-row': {
            'display': 'block',
        },
        'tooltip-label': {
            'display': 'inline-block',
            'fontWeight': 'bold',
        },
        'tooltip-value': {
            'display': 'inline-block',
        },
    }

    # Define divisor styles
    n_divisors = 10
    n_small_divisors = 10
    divisor_width = total_width / n_divisors
    small_divisor_width = 100 / n_small_divisors
    div_s = total_t / n_divisors

    divisor = {
        'position': 'absolute',
        'top': '0',
        'height': '100%',
        'width': '%f%%' % divisor_width,
        'borderRight': 'solid 1px #d4d4d4',

    }
    for i in range(n_divisors):
        divisor = copy.deepcopy(divisor)

        divisor['left'] = '%f%%' % (i*divisor_width,)

        stylesheet['divisor-%d' % i] = divisor

    small_divisor = {
        'position': 'absolute',
        'top': '0',
        'height': '10px',
        'width': '%f%%' % small_divisor_width,
        'borderRight': 'solid 1px #d4d4d4',
    }
    for i in range(n_small_divisors-1):
        small_divisor = copy.deepcopy(small_divisor)

        small_divisor['left'] = '%f%%' % (i*small_divisor_width,)

        stylesheet['small-divisor-%d' % i] = small_divisor

    # Transform profile into events and organise
    runtime_events = OrderedDict()

    for uid, obj in profile.monitored_tessera.items():
        remote_runtime_id = obj['runtime_id']
        if remote_runtime_id not in runtime_events:
            runtime_events[remote_runtime_id] = OrderedDict()

        runtime_events[remote_runtime_id][uid] = {
            'events': [],
            'tasks': OrderedDict(),
        }

        for runtime_id, runtime in obj.proxy_events.items():
            if runtime_id not in runtime_events:
                runtime_events[runtime_id] = OrderedDict()

            runtime_events[runtime_id][uid] = {
                'events': [],
                'tasks': OrderedDict(),
            }

            events = list(runtime.values())
            for index, event in enumerate(events):
                event['uid'] = uid
                event['runtime_id'] = runtime_id
                event['remote_runtime_id'] = remote_runtime_id
                event['event_type'] = 'proxy'
                event['event_t'] = event['event_t'] - start_t
                event['rel_event_t'] = event['event_t']/total_t * 100.

                try:
                    next_event = events[index+1]
                except IndexError:
                    pass
                else:
                    event['end_t'] = next_event['event_t'] - start_t
                    event['total_t'] = event['end_t'] - event['event_t']

                    event['rel_end_t'] = event['end_t']/total_t * 100.
                    event['rel_total_t'] = event['total_t']/total_t * 100.

                runtime_events[runtime_id][uid]['events'].append(event)

        events = list(obj.remote_events.values())
        for index, event in enumerate(events):
            event['uid'] = uid
            event['runtime_id'] = remote_runtime_id
            event['remote_runtime_id'] = obj['runtime_id']
            event['event_type'] = 'remote'
            event['event_t'] = event['event_t'] - start_t
            event['rel_event_t'] = event['event_t'] / total_t * 100.

            try:
                next_event = events[index + 1]
            except IndexError:
                pass
            else:
                event['end_t'] = next_event['event_t'] - start_t
                event['total_t'] = event['end_t'] - event['event_t']

                event['rel_end_t'] = event['end_t'] / total_t * 100.
                event['rel_total_t'] = event['total_t'] / total_t * 100.

            runtime_events[remote_runtime_id][uid]['events'].append(event)

    for uid, obj in profile.monitored_tasks.items():
        remote_runtime_id = obj['runtime_id']
        tessera_id = obj['tessera_id']

        for runtime_id, runtime in obj.proxy_events.items():
            if runtime_id not in runtime_events:
                runtime_events[runtime_id] = OrderedDict()

            runtime_events[runtime_id][tessera_id]['tasks'][uid] = []

            events = list(runtime.values())
            for index, event in enumerate(events):
                event['uid'] = uid
                event['runtime_id'] = runtime_id
                event['remote_runtime_id'] = remote_runtime_id
                event['tessera_id'] = obj['tessera_id']
                event['event_type'] = 'proxy'
                event['event_t'] = event['event_t'] - start_t
                event['rel_event_t'] = event['event_t']/total_t * 100.

                try:
                    next_event = events[index+1]
                except IndexError:
                    pass
                else:
                    event['end_t'] = next_event['event_t'] - start_t
                    event['total_t'] = event['end_t'] - event['event_t']

                    event['rel_end_t'] = event['end_t']/total_t * 100.
                    event['rel_total_t'] = event['total_t']/total_t * 100.

                runtime_events[runtime_id][tessera_id]['tasks'][uid].append(event)

        runtime_events[remote_runtime_id][tessera_id]['tasks'][uid] = []

        events = list(obj.remote_events.values())
        for index, event in enumerate(events):
            event['uid'] = uid
            event['runtime_id'] = remote_runtime_id
            event['remote_runtime_id'] = obj['runtime_id']
            event['tessera_id'] = obj['tessera_id']
            event['event_type'] = 'remote'
            event['event_t'] = event['event_t'] - start_t
            event['rel_event_t'] = event['event_t'] / total_t * 100.

            try:
                next_event = events[index + 1]
            except IndexError:
                pass
            else:
                event['end_t'] = next_event['event_t'] - start_t
                event['total_t'] = event['end_t'] - event['event_t']

                event['rel_end_t'] = event['end_t'] / total_t * 100.
                event['rel_total_t'] = event['total_t'] / total_t * 100.

            runtime_events[remote_runtime_id][tessera_id]['tasks'][uid].append(event)

    for runtime_id, runtime in runtime_events.items():
        for obj_id, obj in runtime.items():
            obj['tasks'] = OrderedDict(sorted(obj['tasks'].items(),
                                              key=lambda x: x[1][0]['event_t']))

    runtime_events = OrderedDict(sorted(runtime_events.items(), key=lambda x: x[0]))

    # Create display elements and ruler
    z_index = [99999]
    display_elements = OrderedDict()

    def generate_element(index, event):
        style = copy.deepcopy(stylesheet['event'])

        style['zIndex'] = str(z_index[-1])
        z_index.append(z_index[-1] - 1)

        style['left'] = '%f%%' % event['rel_event_t']
        style['width'] = '%f%%' % event['rel_total_t']

        colour = cmap(event['rel_total_t'] / 100)
        style['backgroundColor'] = 'rgba(%f, %f, %f, %f)' % (colour[0] * 255, colour[1] * 255, colour[2] * 255, colour[3])

        event_info = {
            'event_uid': 'UID:',
            'uid': 'Object UID:',
            'name': 'Event name:',
            'runtime_id': 'Event runtime:',
            'remote_runtime_id': 'Remote runtime:',
            'event_t': 'Start time (s):',
            'end_t': 'End time (s):',
            'total_t': 'Total time (s):',
            'rel_total_t': 'Total relative time (%):',
        }

        event_tooltip = [
            dash.html.Div(
                [
                    dash.html.Span(value, style=stylesheet['tooltip-label']),
                    dash.html.Span(str(event[key]), style=stylesheet['tooltip-value']),
                ],
                style=stylesheet['tooltip-row'],
            ) for key, value in event_info.items()
        ]

        event_element = dash.html.Div(
            [
                dash.html.Span(event['name'], className='event-label', style=stylesheet['event-label']),
                dash.html.Span(event['uid'], className='event-label', style=stylesheet['event-label']),
                dash.html.Div(
                    event_tooltip,
                    className='tooltip',
                    style=stylesheet['tooltip'],
                ),
            ],
            id=event['event_uid'],
            className='event',
            style=style,
        )

        return event_element

    def generate_elements():
        elements = []

        for runtime_id, runtime in runtime_events.items():
            display_elements[runtime_id] = DisplayElement(
                label_id=runtime_id,
                label=dash.html.Span(runtime_id,
                                     id='%s-label' % runtime_id,
                                     n_clicks=0,
                                     className='runtime-label',
                                     style=stylesheet['runtime-label']),
            )

            for obj_id, obj in runtime.items():
                label_id = '%s-%s' % (runtime_id, obj_id)

                display_elements[obj_id] = DisplayElement(
                    label_id=label_id,
                    label=dash.html.Span(obj_id,
                                         id='%s-label' % label_id,
                                         n_clicks=0,
                                         className='row-label',
                                         style=stylesheet['row-label']),
                )

                for index, event in enumerate(obj['events']):
                    if 'rel_total_t' not in event:
                        continue

                    event_element = generate_element(index, event)
                    display_elements[obj_id].elements.append(event_element)

                for task_id, task in obj['tasks'].items():
                    task_label_id = '%s-%s' % (runtime_id, task_id)

                    display_elements[task_id] = DisplayElement(
                        label_id=task_label_id,
                        label=dash.html.Span(task_id,
                                             id='%s-label' % task_label_id,
                                             n_clicks=0,
                                             className='row-label',
                                             style=stylesheet['row-label']),
                    )

                    for index, event in enumerate(task):
                        if 'rel_total_t' not in event:
                            continue

                        event_element = generate_element(index, event)
                        display_elements[task_id].elements.append(event_element)

                    task_html = dash.html.Div(
                        [display_elements[task_id].label] + display_elements[task_id].elements,
                        id=task_label_id,
                        style={**stylesheet['runtime-row'], **stylesheet['task-row']},
                    )
                    display_elements[task_id].html = task_html

                    display_elements[obj_id].sub_elements[task_id] = display_elements[task_id]

                obj_html = dash.html.Div(
                    [display_elements[obj_id].label] + display_elements[obj_id].elements,
                    id=label_id,
                    style=stylesheet['runtime-row'],
                )
                display_elements[obj_id].html = obj_html

                display_elements[runtime_id].sub_elements[obj_id] = display_elements[obj_id]

            runtime_rows = []

            def generate_rows(obj_id):
                for sub_id, sub_element in display_elements[obj_id].sub_elements.items():
                    runtime_rows.append(sub_element.html)

                    generate_rows(sub_id)

            generate_rows(runtime_id)

            runtime_html = dash.html.Div(
                [display_elements[runtime_id].label] + runtime_rows,
                id=runtime_id,
                style=stylesheet['runtime'],
            )
            display_elements[runtime_id].html = runtime_html
            elements.append(runtime_html)

            def generate_callback(obj_id):

                label_id = display_elements[obj_id].label_id
                output_labels = [each.label_id
                                 for each in display_elements[obj_id].sub_elements.values()]

                if not len(output_labels):
                    return

                @app.callback(
                    [Output(each, 'style') for each in output_labels],
                    [Input('%s-label' % label_id, 'n_clicks')],
                    [State(each, 'style') for each in output_labels],
                    prevent_initial_call=True,
                )
                def clicks(n_clicks, *styles):
                    closed = (n_clicks % 2) != 0
                    display_elements[obj_id].label.n_clicks += 1

                    if closed:
                        for style in styles:
                            style['display'] = 'block'

                    else:
                        for style in styles:
                            style['display'] = 'none'

                    return styles

                for sub_id, sub_element in display_elements[obj_id].sub_elements.items():
                    generate_callback(sub_id)

                return clicks

            generate_callback(runtime_id)

        return elements

    def generate_ruler():
        divisors = []

        for i in range(n_divisors):
            small_divisors = [
                dash.html.Span('%.2f s' % (div_s*i,), style=stylesheet['divisor-label'])
            ]

            for j in range(n_small_divisors-1):
                small_divisor = dash.html.I(style=stylesheet['small-divisor-%d' % j])
                small_divisors.append(small_divisor)

            divisor = dash.html.I(small_divisors, className='divisor', style=stylesheet['divisor-%d' % i])
            divisors.append(divisor)

        return divisors

    # Configure the app
    app.config['suppress_callback_exceptions'] = True

    app.layout = dash.html.Div(
        [
            dash.html.Div(
                [],
                id='sidebar',
                style=stylesheet['sidebar'],
            ),
            dash.html.Div(
                dash.html.Div(
                    generate_ruler() + generate_elements(),
                    id='inner-main',
                    style=stylesheet['inner-main'],
                ),
                id='main',
                style=stylesheet['main'],
            ),
        ],
        id='content',
        style=stylesheet['content'],
    )

    # Start server
    app.run_server(debug=True)


if __name__ == '__main__':
    go(auto_envvar_prefix='MOSAIC')
