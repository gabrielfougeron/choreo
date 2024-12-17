{{ fullname | escape | underline}}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}

   {% block methods %}

   .. rubric:: {{ _('Class Constructor') }}
   .. automethod:: __init__
      :no-index:

   {% if methods %}
   .. rubric:: {{ _('Methods Summary') }}

   .. autosummary::
    :nosignatures:
    :recursive:
    :toctree:

   {% for item in methods %}
      ~{{ name }}.{{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}

   {% block attributes %}
   {% if attributes %}
   .. rubric:: {{ _('Attributes') }}

   .. autosummary::
    :toctree:
   {% for item in attributes %}
      ~{{ name }}.{{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}

