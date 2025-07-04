from flask import Flask, Response


class Gateway(Flask):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.add_url_rule("hello", "hello", self.hello, methods=["GET"])

    def hello(self):
        return Response("Hello, World!", status=200, mimetype="text/plain")
