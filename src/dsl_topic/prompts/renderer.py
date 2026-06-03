import os
import jinja2


class JinjaTemplateManager:
    """Loads and renders the Jinja prompt templates shipped in this package."""

    def __init__(self, template_dir: str = None):
        if template_dir is None:
            template_dir = os.path.join(os.path.dirname(__file__), 'templates')
        self.env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(template_dir))

    def render(self, template_name: str, **kwargs) -> str:
        """Render a template with the given keyword arguments."""
        template = self.env.get_template(template_name)
        return template.render(**kwargs)


jinja_template_manager = JinjaTemplateManager()