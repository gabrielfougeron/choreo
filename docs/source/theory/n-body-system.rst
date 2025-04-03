.. _n-body-system:


What is a n-body system ?
=========================

A n-body system is a set of :math:`n` classical **non-relativistic** point masses undergoing **binary** interactions. These interactions are modelled by forces following a direction **aligned** with the straight line joining the bodies positions, and whose magnitude only depends on the **inter-body distance**.

In :mod:`choreo`, n-body systems are represented by the class :class:`choreo.NBodySyst`.

.. note:: Dimensionality of the system (geodim)

The following sections detail three different formulations of the evolution equations of n-body systems.

.. _newtonian-pov:

The Newtonian point of view
---------------------------

The mass of a body :math:`i = 0 \dots n-1` is denoted :math:`m_i > 0`, its position :math:`x_i`, and the resultant of all forces applied :math:`f_i`.
The Newtonian formulation of the equations of motion of a n-body system states that the mass times the acceleration of a body equals the forces applied to this body:

.. math:: m_i \frac{\dd^2 x_i}{\dd t^2} = f_i
    :label: Newton

This equation alone specifies the evolution of the unknown variables, *i.e.* the positions :math:`\mathbf{q}(t) \eqdef (x_0(t), \dots, x_{n-1}(t))`.

.. note:: Although the usage of the letter :math:`\mathbf{q}` here is confusing, it follows standard notation in the study of Hamiltonian and Lagrangian systems. It should not be mistaken for the *charges* that we denote :math:`q` in this documentation.

The force :math:`f_i` in the above equation is the resultant of binary interactions with all the other bodies in the system.
Moreover, the force that body :math:`i` applies on body :math:`j` is the opposite of the force that body :math:`j` applies on body :math:`i`. It is pointed in the direction of the other body, and its magnitude is a function of the inter-body distance :math:`\|x_j = x_i\|`. These restrictions constraint the forces in the system to be conservative:

.. math::

    f_i &= \sum_{j \neq i} f_{i,j} \\
    f_{i,j} &= -f_{j,i} = \pm \|f_{i,j}(\|x_i - x_j\|)\| \frac{x_i - x_j}{\|x_i - x_j\|}\\
    &= - \frac{\partial}{\partial x_i}V_{i,j}(\|x_i - x_j\|)\\

For both performance reasons and ease of input, :mod:`choreo` additionally constraints the inter-body potential :math:`V_{i,j}` to be proportional to a **universal** user-specified potential denoted :math:`V_{\mathrm{u}}`, with body-specific proportionality constants :math:`q_i` and :math:`q_j` called *charges*:

.. math::

    V_{i,j} = q_i q_j V_{\mathrm{u}}(\|x_i - x_j\|)
    
This convention is compatible with both the classical Newtonian gravitational potential (with :math:`m_i = q_i`) and the electrostatic potential (in which case the body charges are the electric charges) if the potential is proportional to the inverse of the distance:

.. math::

    V(x) \propto \frac{1}{x}

.. note:: Independence wrt the proportionality constant for homogeneous potentials.

While equation :eq:`Newton` as it is could be directly plugged in a `general purpose ODE solver <https://docs.scipy.org/doc/scipy/reference/integrate.html#solving-initial-value-problems-for-ode-systems>`_ to get an approximate solution, the next two sections highlight how variations on the problem formulation can be exploited for more performance, and to better deal with :ref:`periodicity constraints<periodicity>`.

.. _hamiltonian-pov:

The Hamiltonian point of view
-----------------------------

While Newton's law of motion :eq:`Newton` completely determines the evolution of a n-body system, their Hamiltonian reformulation reveal hidden mathematical structure that can be exploited to find more precise approximate solutions at a lower computational cost.

The unknown **independent** variables in the Hamiltonian formulations are both the positions :math:`\mathbf{q}(t) = (x_0(t), \dots, x_{n-1}(t))`, and the momenta :math:`\mathbf{p}(t) \eqdef (p_0(t), \dots, p_{n-1}(t))`. Given a *scalar* function of the momenta and positions denoted :math:`H(\mathbf{q}, \mathbf{p})` and called the *Hamiltonian* of the system, the equations of motion for the evolution of the inpendant variables read:

.. math:: \frac{\dd \mathbf{q}}{\dd t}  &= \frac{\partial H}{\partial \mathbf{p}} \\
    \frac{\dd \mathbf{p}}{\dd t}  &= -\frac{\partial H}{\partial \mathbf{q}}  \\
    :label: Hamilton_eq_evolution

The Newton equations of motion :eq:`Newton` are retrieved for the following choice of Hamiltonian:

.. math::
    H(\mathbf{q}, \mathbf{p}) &= T(\mathbf{p}) + V(\mathbf{q})  \\
    &= \sum_{i=0}^{n-1} \frac{p_i^2}{2 m_i} + \sum_{i=0}^{n-1} \sum_{j\neq i} q_i q_j V(\|x_i - x_j\|)\\
    :label: Hamiltonian_of_nbodysyst

This particular Hamiltonian is called **partionned** since it decomposes into the sum of a **kinetic energy** :math:`T(\mathbf{p})` that is a function of the momenta *only*, and a **potential energy** :math:`V(\mathbf{q})` that is a function of the positions *only*. This partitioned structure is exploited in the ODE RK methods ref ???



.. note:: TODO : Lagrangian least action principle vs Hamiltonian least action principle. Stability ?



The Lagrangian point of view
----------------------------

.. _periodicity:

What is periodicity ?
=====================

