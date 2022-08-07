Ordinary differential equations in choreo
=========================================

.. _ode_FOS:

First order systems
-------------------

First order ordinary equations have the form:

.. math::
    \frac{\mathrm{d} x(t)}{\mathrm{d}t}  = f(t,x(t))


.. _ode_HS:

Hamiltonian systems
-------------------
A Hamiltonian system is a specific case of :ref:`first order system of ordinary differential equations<ode_FOS>`.
Given a sufficiently regular function :math:`H(p,q)` called the **Hamiltonian**, a Hamiltonian system reads:

.. math::
    \frac{\mathrm{d} q(t)}{\mathrm{d}t}  &= \frac{\partial H}{\partial p}(q(t),p(t)) \\
    \frac{\mathrm{d} p(t)}{\mathrm{d}t}  &= -\frac{\partial H}{\partial q}(q(t),p(t))  \\

.. _ode_PHS:

Partitioned Hamiltonian systems
-------------------------------

If the Hamiltonian of the system has the form :math:`H(p,q) = T(p) + V(q)`, then the system is called a **partitioned** Hamiltonian system. This specific case is amenable to simulation optimizations.

Example: a system of :math:`n` classical interacting point masses interacting under Newtonian gravity is a partitioned Hamiltonian system, where:

.. math::
    T(p)  &= \sum_{i=1}^{n} \frac{p_i^2}{m_i} \\
    V(q)  &= \sum_{i=1}^{n-1} \sum_{j=i+1}^{n} G \frac{m_i m_j}{|q_i-q_j|}  \\

