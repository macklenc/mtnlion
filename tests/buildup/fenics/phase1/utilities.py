import fenics as fem
import matplotlib.pyplot as plt
import numpy as np

import mtnlion.comsol as comsol
import mtnlion.engine as engine


def gather_data():
    # Load required cell data
    resources = '../../../reference/'
    params = engine.fetch_params(resources + 'GuAndWang_parameter_list.xlsx')
    d_comsol = comsol.load(resources + 'guwang2.npz')
    return d_comsol, params


def mkparam(markers, k_1=0, k_2=0, k_3=0, k_4=0):
    cppcode = """
    class K : public Expression
    {
        public:
            void eval(Array<double>& values,
                      const Array<double>& x,
                      const ufc::cell& cell) const
            {
                switch((*markers)[cell.index]){
                case 1:
                    values[0] = k_1;
                    break;
                case 2:
                    values[0] = k_2;
                    break;
                case 3:
                    values[0] = k_3;
                    break;
                case 4:
                    values[0] = k_4;
                    break;
                default:
                    values[0] = 0;
                }
            }

        std::shared_ptr<MeshFunction<std::size_t>> markers;
        double k_1, k_2, k_3, k_4;
    };
    """

    var = fem.Expression(cppcode=cppcode, degree=1)
    var.markers = markers
    var.k_1, var.k_2, var.k_3, var.k_4 = k_1, k_2, k_3, k_4

    return var


def overlay_plt(xdata, time, title, *ydata, figsize=(15, 9), linestyles=('-', '--')):
    fig, ax = plt.subplots(figsize=figsize)

    new_x = np.repeat([xdata], len(time), axis=0).T

    for i, data in enumerate(ydata):
        if i is 1:
            plt.plot(new_x, data.T, linestyles[i], marker='o')
        else:
            plt.plot(new_x, data.T, linestyles[i])
        plt.gca().set_prop_cycle(None)
    plt.grid(), plt.title(title)

    legend1 = plt.legend(['t = {}'.format(t) for t in time], title='Time', bbox_to_anchor=(1.01, 1), loc=2,
                         borderaxespad=0.)
    ax.add_artist(legend1)

    h = [plt.plot([], [], color="gray", ls=linestyles[i])[0] for i in range(len(linestyles))]
    plt.legend(handles=h, labels=["FEniCS", "COMSOL"], title="Solver", bbox_to_anchor=(1.01, 0), loc=3,
               borderaxespad=0.)
