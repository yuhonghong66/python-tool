from django.conf.urls import url
from . import views


app_name = 'blog'
urlpatterns = [
    # NewBlogView
    url(r'^new/', views.NewBlogView.as_view(), name='new-blog'),
    url(r'^post/new', views.NewBlogPostView.as_view(), name='new-blog-post'),
    url(r'^(?P<pk>\d+)/update/$', views.UpdateBlogView.as_view(), name='update-blog'),
    url(r'^post/(?P<pk>\d+)/update/$', views.UpdateBlogPostView.as_view(), name='update-blog-post'),
    url(r'^post/(?P<pk>\d+)$', views.BlogPostDetailsView.as_view(), name='blog-post-details'),
    url(r'^post/(?P<pk>\d+)/share/$', views.ShareBlogPostView.as_view(), name='share-blog-post-with-blog'),
    url(r'^post/(?P<post_pk>\d+)/share/to/(?P<blog_pk>\d+)/$', views.SharePostWithBlog.as_view(),
        name='share-post-with-blog'),
    url(r'^post/(?P<post_pk>\d+)/stop/share/to/(?P<blog_pk>\d+)/$', views.StopSharingPostWithBlog.as_view(),
        name='stop-sharing-post-with-blog'),
]